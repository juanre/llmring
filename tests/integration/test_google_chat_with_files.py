# ABOUTME: Integration tests for Google chat with files.
# ABOUTME: Tests chat request with uploaded file using File API.
import os

import pytest

from llmring.providers.google_api import GoogleProvider
from llmring.schemas import Message


@pytest.mark.skipif(
    not (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GEMINI_API_KEY")
    ),
    reason="GOOGLE_API_KEY, GEMINI_API_KEY, or GOOGLE_GEMINI_API_KEY not set",
)
@pytest.mark.asyncio
async def test_chat_with_uploaded_file():
    """Test chat request with uploaded file using File API."""
    provider = GoogleProvider()

    try:
        # Upload a file
        upload_response = await provider.upload_file(
            file="tests/fixtures/sample.txt", purpose="analysis"
        )

        # Use file in chat
        messages = [Message(role="user", content="What is in this file?")]
        response = await provider.chat(
            messages=messages, model="gemini-2.5-flash", files=[upload_response.file_id]
        )

        # Verify response
        assert response.content is not None
        assert len(response.content) > 0
        assert response.model

        # Cleanup
        await provider.delete_file(upload_response.file_id)
    finally:
        await provider.aclose()
