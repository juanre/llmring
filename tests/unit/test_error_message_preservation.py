import asyncio
import pytest

from llmring.net.retry import RetryError
from llmring.providers.google_api import GoogleProvider


@pytest.mark.asyncio
async def test_retryerror_preserves_root_message():
    timeout = asyncio.TimeoutError("Request timed out after 60 seconds")
    # Simulate empty message RetryError with cause
    err = RetryError("")
    err.__cause__ = timeout

    # With our reform, RetryError with empty message defaults to "Retry failed"
    assert str(err) == "Retry failed"

    # But we can access the original via __cause__
    assert str(err.__cause__) == "Request timed out after 60 seconds"


def test_google_provider_extract_error_message_from_chain(monkeypatch):
    provider = GoogleProvider(api_key="test")

    # Build nested exceptions: RetryError("") caused by TimeoutError("...")
    timeout = asyncio.TimeoutError("Request timed out after 60 seconds")
    retry = RetryError("")
    retry.__cause__ = timeout

    msg = provider._extract_error_message(retry)
    assert msg == "Request timed out"


def test_google_provider_handles_cancelled_error(monkeypatch):
    provider = GoogleProvider(api_key="test")

    cancelled = asyncio.CancelledError()
    retry = RetryError("unknown error")
    retry.__cause__ = cancelled

    msg = provider._extract_error_message(retry)
    assert "cancelled" in msg.lower()


