"""Integration tests for streaming functionality."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from llmring.schemas import LLMRequest, Message, StreamChunk
from llmring.service import LLMRing


@pytest.mark.asyncio
async def test_streaming_response_structure():
    """Test that streaming responses have the correct structure."""
    test_lockfile = Path(__file__).parent / "llmring.lock.json"
    service = LLMRing(lockfile_path=str(test_lockfile))

    # Mock the provider to avoid real API calls
    mock_provider = AsyncMock()

    # Create mock stream chunks
    async def mock_stream():
        chunks = [
            StreamChunk(delta="Hello", model="test-model"),
            StreamChunk(delta=" world", model="test-model"),
            StreamChunk(
                delta="!",
                model="test-model",
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            ),
        ]
        for chunk in chunks:
            yield chunk

    mock_provider.chat = AsyncMock(return_value=mock_stream())

    # Replace provider in the service
    service.providers["test"] = mock_provider

    request = LLMRequest(
        model="test:model", messages=[Message(role="user", content="Test")], stream=True
    )

    # Collect stream chunks
    chunks = []
    stream = await service.chat(request)
    async for chunk in stream:
        chunks.append(chunk)

    # Verify chunks
    assert len(chunks) == 3
    assert chunks[0].delta == "Hello"
    assert chunks[1].delta == " world"
    assert chunks[2].delta == "!"
    assert chunks[2].finish_reason == "stop"
    assert chunks[2].usage is not None


@pytest.mark.asyncio
async def test_streaming_accumulates_content():
    """Test that streaming properly accumulates content."""
    test_lockfile = Path(__file__).parent / "llmring.lock.json"
    service = LLMRing(lockfile_path=str(test_lockfile))

    mock_provider = AsyncMock()

    async def mock_stream():
        parts = ["The", " quick", " brown", " fox"]
        for part in parts:
            yield StreamChunk(delta=part, model="test-model")

    mock_provider.chat = AsyncMock(return_value=mock_stream())
    service.providers["test"] = mock_provider

    request = LLMRequest(
        model="test:model", messages=[Message(role="user", content="Test")], stream=True
    )

    # Accumulate content
    full_content = ""
    stream = await service.chat(request)
    async for chunk in stream:
        if chunk.delta:
            full_content += chunk.delta

    assert full_content == "The quick brown fox"


@pytest.mark.asyncio
async def test_non_streaming_returns_complete_response():
    """Test that non-streaming returns a complete response."""
    from llmring.schemas import LLMResponse

    test_lockfile = Path(__file__).parent / "llmring.lock.json"
    service = LLMRing(lockfile_path=str(test_lockfile))

    mock_provider = AsyncMock()
    mock_response = LLMResponse(
        content="Complete response",
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        finish_reason="stop",
    )

    mock_provider.chat = AsyncMock(return_value=mock_response)
    service.providers["test"] = mock_provider

    request = LLMRequest(
        model="test:model",
        messages=[Message(role="user", content="Test")],
        stream=False,
    )

    response = await service.chat(request)

    assert isinstance(response, LLMResponse)
    assert response.content == "Complete response"
    assert response.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_streaming_with_error_handling():
    """Test that streaming handles errors gracefully."""
    test_lockfile = Path(__file__).parent / "llmring.lock.json"
    service = LLMRing(lockfile_path=str(test_lockfile))

    mock_provider = AsyncMock()

    async def mock_stream_with_error():
        yield StreamChunk(delta="Start", model="test-model")
        raise Exception("Stream interrupted")

    mock_provider.chat = AsyncMock(return_value=mock_stream_with_error())
    service.providers["test"] = mock_provider

    request = LLMRequest(
        model="test:model", messages=[Message(role="user", content="Test")], stream=True
    )

    chunks = []
    stream = await service.chat(request)

    with pytest.raises(Exception, match="Stream interrupted"):
        async for chunk in stream:
            chunks.append(chunk)

    # Should have received at least the first chunk
    assert len(chunks) == 1
    assert chunks[0].delta == "Start"


@pytest.mark.asyncio
async def test_streaming_preserves_model_info():
    """Test that streaming preserves model information in chunks."""
    test_lockfile = Path(__file__).parent / "llmring.lock.json"
    service = LLMRing(lockfile_path=str(test_lockfile))

    mock_provider = AsyncMock()

    async def mock_stream():
        yield StreamChunk(delta="Test", model="fast")
        yield StreamChunk(delta=" response", model="fast", finish_reason="stop")

    mock_provider.chat = AsyncMock(return_value=mock_stream())
    service.providers["openai"] = mock_provider

    request = LLMRequest(
        model="fast",
        messages=[Message(role="user", content="Test")],
        stream=True,
    )

    chunks = []
    stream = await service.chat(request)
    async for chunk in stream:
        chunks.append(chunk)

    # All chunks should have model info
    for chunk in chunks:
        assert chunk.model == "fast"


@pytest.mark.asyncio
async def test_cli_streaming_format():
    """Test that CLI streaming format works correctly."""
    # This would be better as an actual CLI test, but we can simulate
    from io import StringIO

    test_lockfile = Path(__file__).parent / "llmring.lock.json"
    service = LLMRing(lockfile_path=str(test_lockfile))

    mock_provider = AsyncMock()

    async def mock_stream():
        words = ["Hello", " there", "!", "\n", "How", " are", " you", "?"]
        for word in words:
            yield StreamChunk(delta=word, model="test-model")
            await asyncio.sleep(0.01)  # Simulate delay

    mock_provider.chat = AsyncMock(return_value=mock_stream())
    service.providers["test"] = mock_provider

    request = LLMRequest(
        model="test:model",
        messages=[Message(role="user", content="Greet me")],
        stream=True,
    )

    # Capture output
    output = StringIO()
    stream = await service.chat(request)

    async for chunk in stream:
        if chunk.delta:
            output.write(chunk.delta)
            output.flush()

    result = output.getvalue()
    assert result == "Hello there!\nHow are you?"


@pytest.mark.asyncio
async def test_streaming_with_tools():
    """Test streaming with tool/function calls."""
    test_lockfile = Path(__file__).parent / "llmring.lock.json"
    service = LLMRing(lockfile_path=str(test_lockfile))

    mock_provider = AsyncMock()

    async def mock_stream():
        # Simulate a tool call response
        yield StreamChunk(delta="I'll help you with that.", model="test-model")
        yield StreamChunk(
            delta="",
            model="test-model",
            finish_reason="tool_calls",
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        )

    mock_provider.chat = AsyncMock(return_value=mock_stream())
    service.providers["test"] = mock_provider

    request = LLMRequest(
        model="test:model",
        messages=[Message(role="user", content="Calculate something")],
        tools=[{"type": "function", "function": {"name": "calculate", "parameters": {}}}],
        stream=True,
    )

    chunks = []
    stream = await service.chat(request)
    async for chunk in stream:
        chunks.append(chunk)

    # Should have received chunks including tool call indicator
    assert len(chunks) == 2
    assert chunks[-1].finish_reason == "tool_calls"
    assert chunks[-1].usage is not None
