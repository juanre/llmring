"""Integration tests for streaming functionality."""

import os
from pathlib import Path

import pytest

from llmring.schemas import LLMRequest, LLMResponse, Message
from llmring.service import LLMRing


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not available")
async def test_streaming_response_structure():
    """Test that streaming responses have the correct structure."""
    test_lockfile = Path(__file__).parent / "llmring.lock.json"
    service = LLMRing(lockfile_path=str(test_lockfile))

    request = LLMRequest(
        model="anthropic_fast",
        messages=[Message(role="user", content="Say 'Hello world!' exactly.")],
        max_tokens=20,
    )

    # Collect stream chunks
    chunks = []
    async for chunk in service.chat_stream(request):
        chunks.append(chunk)

    # Verify chunks
    assert len(chunks) > 0, "Should receive at least one chunk"

    # Check that we got content
    content_chunks = [c for c in chunks if c.delta]
    assert len(content_chunks) > 0, "Should have content chunks"

    # Last chunk should have finish_reason
    assert chunks[-1].finish_reason is not None, "Last chunk should have finish_reason"

    # Last chunk should have usage info
    assert chunks[-1].usage is not None, "Last chunk should have usage info"


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not available")
async def test_streaming_accumulates_content():
    """Test that streaming properly accumulates content."""
    test_lockfile = Path(__file__).parent / "llmring.lock.json"
    service = LLMRing(lockfile_path=str(test_lockfile))

    request = LLMRequest(
        model="anthropic_fast",
        messages=[Message(role="user", content="Say exactly: Hello")],
        max_tokens=10,
    )

    # Accumulate content
    full_content = ""
    async for chunk in service.chat_stream(request):
        if chunk.delta:
            full_content += chunk.delta

    assert len(full_content) > 0, "Should have accumulated some content"
    assert "hello" in full_content.lower(), "Content should contain 'hello'"


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not available")
async def test_non_streaming_returns_complete_response():
    """Test that non-streaming returns a complete response."""
    test_lockfile = Path(__file__).parent / "llmring.lock.json"
    service = LLMRing(lockfile_path=str(test_lockfile))

    request = LLMRequest(
        model="anthropic_fast",
        messages=[Message(role="user", content="Say: Hello")],
        max_tokens=10,
    )

    response = await service.chat(request)

    assert isinstance(response, LLMResponse)
    assert len(response.content) > 0, "Should have content"
    assert response.usage is not None, "Should have usage info"
    assert response.usage.get("total_tokens", 0) > 0, "Should have token count"


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not available")
async def test_streaming_preserves_model_info():
    """Test that streaming preserves model information in chunks."""
    test_lockfile = Path(__file__).parent / "llmring.lock.json"
    service = LLMRing(lockfile_path=str(test_lockfile))

    request = LLMRequest(
        model="anthropic_fast",
        messages=[Message(role="user", content="Hi")],
        max_tokens=10,
    )

    chunks = []
    async for chunk in service.chat_stream(request):
        chunks.append(chunk)

    # All chunks should have model info
    for chunk in chunks:
        assert chunk.model is not None, "Each chunk should have model info"
        assert (
            "anthropic" in chunk.model or "claude" in chunk.model.lower()
        ), "Model should be anthropic/claude"
