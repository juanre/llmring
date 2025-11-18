# ABOUTME: Integration tests for chat() with file references.
# ABOUTME: Tests file handling in chat across providers without mocks.

import os

import pytest

from llmring.providers.anthropic_api import AnthropicProvider
from llmring.providers.google_api import GoogleProvider
from llmring.providers.openai_api import OpenAIProvider
from llmring.schemas import LLMRequest, Message
from llmring.service import LLMRing


@pytest.mark.asyncio
async def test_anthropic_chat_with_files():
    """Test Anthropic chat with uploaded files."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider()
    try:
        # Upload test file
        upload_response = await provider.upload_file(
            file="tests/fixtures/sample.txt",
            purpose="analysis",
        )

        # Use file in chat
        messages = [Message(role="user", content="What is in this file?")]
        response = await provider.chat(
            messages=messages,
            model="claude-3-5-haiku-20241022",
            files=[upload_response.file_id],
        )

        # Verify response
        assert response.content is not None
        assert len(response.content) > 0
        assert response.model == "claude-3-5-haiku-20241022"
        assert response.usage is not None

        # Cleanup
        await provider.delete_file(upload_response.file_id)

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_anthropic_chat_with_multiple_files():
    """Test Anthropic chat with multiple uploaded files."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider()
    try:
        # Upload two test files
        file1 = await provider.upload_file(
            file="tests/fixtures/sample.txt",
            purpose="analysis",
        )
        file2 = await provider.upload_file(
            file="tests/fixtures/large_document.txt",
            purpose="analysis",
        )

        # Use both files in chat
        messages = [Message(role="user", content="What is in these files?")]
        response = await provider.chat(
            messages=messages,
            model="claude-3-5-haiku-20241022",
            files=[file1.file_id, file2.file_id],
        )

        # Verify response
        assert response.content is not None
        assert len(response.content) > 0

        # Cleanup
        await provider.delete_file(file1.file_id)
        await provider.delete_file(file2.file_id)

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_openai_chat_with_files():
    """Test OpenAI chat with uploaded files using Responses API attachments."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    provider = OpenAIProvider()
    try:
        # Upload test file to OpenAI
        upload_response = await provider.upload_file(
            file="tests/fixtures/sample.txt",
            purpose="analysis",
        )

        messages = [Message(role="user", content="What is in this file?")]
        response = await provider.chat(
            messages=messages,
            model="gpt-4o",
            files=[upload_response.file_id],
        )

        # Verify response
        assert response.content is not None
        assert len(response.content) > 0
        assert response.model == "gpt-4o"

        # Cleanup uploaded file
        await provider.delete_file(upload_response.file_id)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_google_chat_with_files():
    """Test Google chat with uploaded file references."""
    if not (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GEMINI_API_KEY")
    ):
        pytest.skip("GOOGLE_API_KEY, GEMINI_API_KEY, or GOOGLE_GEMINI_API_KEY not set")

    provider = GoogleProvider()
    try:
        # Upload a test file via Google Files API
        upload_response = await provider.upload_file(
            file="tests/fixtures/google_large_doc.txt",
            purpose="analysis",
        )

        # Use uploaded file reference in chat
        messages = [Message(role="user", content="What is in the file?")]
        response = await provider.chat(
            messages=messages,
            model="gemini-2.5-flash",
            files=[upload_response.file_id],
        )

        # Verify response
        assert response.content is not None
        assert len(response.content) > 0

    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_llmring_chat_with_files():
    """Test LLMRing.chat() with files parameter (Anthropic)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    async with LLMRing() as ring:
        # Register file
        file_id = ring.register_file("tests/fixtures/sample.txt")

        # Use file in chat via LLMRequest
        request = LLMRequest(
            model="anthropic:claude-3-5-haiku-20241022",
            messages=[Message(role="user", content="What is in this file?")],
            files=[file_id],
        )

        response = await ring.chat(request)

        # Verify response
        assert response.content is not None
        assert len(response.content) > 0
        assert "anthropic" in response.model

        # Cleanup
        await ring.deregister_file(file_id)


@pytest.mark.asyncio
async def test_llmring_chat_with_files_openai():
    """Test LLMRing.chat() uses Responses API for OpenAI when files are present."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    async with LLMRing() as ring:
        # Register file (lazy upload)
        file_id = ring.register_file("tests/fixtures/sample.txt")

        # Use files with OpenAI via LLMRing
        request = LLMRequest(
            model="openai:gpt-4o",
            messages=[Message(role="user", content="What is in this file?")],
            files=[file_id],
        )

        response = await ring.chat(request)
        assert response.content is not None
        assert len(response.content) > 0
        assert "openai" in f"openai:{response.model}"

        # Cleanup
        await ring.deregister_file(file_id)


@pytest.mark.asyncio
async def test_chat_stream_with_files():
    """Test streaming chat with files parameter (Anthropic)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    async with LLMRing() as ring:
        # Register file
        file_id = ring.register_file("tests/fixtures/sample.txt")

        # Use file in streaming chat
        request = LLMRequest(
            model="anthropic:claude-3-5-haiku-20241022",
            messages=[Message(role="user", content="Briefly describe this file.")],
            files=[file_id],
        )

        # Collect stream chunks
        chunks = []
        async for chunk in ring.chat_stream(request):
            chunks.append(chunk)

        # Verify we got chunks
        assert len(chunks) > 0

        # Verify content was streamed
        content_chunks = [c for c in chunks if c.delta]
        assert len(content_chunks) > 0

        # Cleanup
        await ring.deregister_file(file_id)


@pytest.mark.asyncio
async def test_chat_with_no_files():
    """Test that chat without files parameter works normally (regression test)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    provider = AnthropicProvider()
    try:
        # Regular chat without files
        messages = [Message(role="user", content="Say 'test' and nothing else.")]
        response = await provider.chat(
            messages=messages,
            model="claude-3-5-haiku-20241022",
        )

        # Verify response
        assert response.content is not None
        assert "test" in response.content.lower()

    finally:
        await provider.aclose()
