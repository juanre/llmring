"""Tests for server-side usage logging functionality.

These tests use real llmring-server (via fixture) and real provider API calls.
"""

import os

import pytest

from llmring.schemas import LLMRequest, Message
from llmring.service import LLMRing

# Check if llmring-server is properly installed (not just namespace package)
try:
    from llmring_server.main import app as _test_import  # noqa: F401

    LLMRING_SERVER_INSTALLED = True
except ImportError:
    LLMRING_SERVER_INSTALLED = False


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMRING_SERVER_INSTALLED, reason="llmring-server not properly installed")
async def test_llmring_with_server_logs_usage(llmring_server_client, project_headers):
    """Test that LLMRing logs usage to server when configured."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    # Create LLMRing connected to the test server
    ring = LLMRing(
        origin="test-usage-logging",
        server_url="http://test",  # Will be overridden by injecting client
        api_key="proj_test",
    )
    # Inject the test server client directly
    ring.server_client.client = llmring_server_client

    # Make a real chat request
    request = LLMRequest(
        model="openai:gpt-4o-mini",
        messages=[Message(role="user", content="Say 'test' and nothing else.")],
        max_tokens=10,
    )

    response = await ring.chat(request)

    # Verify we got a real response
    assert response.content is not None
    assert len(response.content) > 0
    assert response.usage["prompt_tokens"] > 0
    assert response.usage["completion_tokens"] > 0

    # Query the server to verify the log was stored
    logs_response = await llmring_server_client.get(
        "/api/v1/logs",
        params={"origin": "test-usage-logging", "limit": 1},
        headers=project_headers,
    )
    assert logs_response.status_code == 200

    logs = logs_response.json()
    assert len(logs) > 0

    log_entry = logs[0]
    assert log_entry["provider"] == "openai"
    assert "gpt-4o-mini" in log_entry["model"]
    assert log_entry["input_tokens"] > 0
    assert log_entry["output_tokens"] > 0
    assert log_entry["origin"] == "test-usage-logging"


@pytest.mark.asyncio
async def test_llmring_without_server_no_logging():
    """Test that LLMRing without server doesn't attempt logging."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    # Create LLMRing without server
    ring = LLMRing(origin="test")

    # Verify server_client is None
    assert ring.server_client is None

    # Make a real chat request - should work without server
    request = LLMRequest(
        model="openai:gpt-4o-mini",
        messages=[Message(role="user", content="Say 'test' and nothing else.")],
        max_tokens=10,
    )

    response = await ring.chat(request)

    # Verify response works without server
    assert response.content is not None
    assert len(response.content) > 0


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMRING_SERVER_INSTALLED, reason="llmring-server not properly installed")
async def test_usage_logging_with_alias(llmring_server_client, project_headers, tmp_path):
    """Test that usage logging includes alias information."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    # Create a test lockfile with an alias
    from llmring.lockfile_core import Lockfile

    lockfile_path = tmp_path / "llmring.lock"
    lockfile = Lockfile.create_default()
    lockfile.set_binding("test-alias", "openai:gpt-4o-mini")
    lockfile.save(lockfile_path)

    # Create LLMRing with the test lockfile
    ring = LLMRing(
        origin="test-alias-logging",
        server_url="http://test",
        api_key="proj_test",
        lockfile_path=str(lockfile_path),
    )
    # Inject the test server client directly
    ring.server_client.client = llmring_server_client

    # Make a chat request using the alias
    request = LLMRequest(
        model="test-alias",
        messages=[Message(role="user", content="Say 'test' and nothing else.")],
        max_tokens=10,
    )

    await ring.chat(request)

    # Query the server to verify the log includes alias
    logs_response = await llmring_server_client.get(
        "/api/v1/logs",
        params={"origin": "test-alias-logging", "limit": 1},
        headers=project_headers,
    )
    assert logs_response.status_code == 200

    logs = logs_response.json()
    assert len(logs) > 0

    log_entry = logs[0]
    assert log_entry["alias"] == "test-alias"
    assert log_entry["provider"] == "openai"
    assert "gpt-4o-mini" in log_entry["model"]


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMRING_SERVER_INSTALLED, reason="llmring-server not properly installed")
async def test_streaming_usage_logging(llmring_server_client, project_headers):
    """Test that streaming responses also log usage."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    # Create LLMRing connected to the test server
    ring = LLMRing(
        origin="test-streaming-logging",
        server_url="http://test",
        api_key="proj_test",
    )
    # Inject the test server client directly
    ring.server_client.client = llmring_server_client

    # Make a streaming request
    request = LLMRequest(
        model="openai:gpt-4o-mini",
        messages=[Message(role="user", content="Say 'hello world'.")],
        max_tokens=20,
    )

    # Consume the stream
    chunks = []
    async for chunk in ring.chat_stream(request):
        chunks.append(chunk)

    # Verify we got chunks
    assert len(chunks) > 0

    # Wait for async logging to complete
    import asyncio
    await asyncio.sleep(0.3)

    # Query the server to verify the log was stored
    logs_response = await llmring_server_client.get(
        "/api/v1/logs",
        params={"origin": "test-streaming-logging", "limit": 1},
        headers=project_headers,
    )
    assert logs_response.status_code == 200

    logs = logs_response.json()
    assert len(logs) > 0

    log_entry = logs[0]
    assert log_entry["provider"] == "openai"
    assert log_entry["input_tokens"] > 0
    assert log_entry["output_tokens"] > 0


@pytest.mark.asyncio
@pytest.mark.skipif(not LLMRING_SERVER_INSTALLED, reason="llmring-server not properly installed")
async def test_anthropic_usage_logging(llmring_server_client, project_headers):
    """Test usage logging with Anthropic provider."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    # Create LLMRing connected to the test server
    ring = LLMRing(
        origin="test-anthropic-logging",
        server_url="http://test",
        api_key="proj_test",
    )
    # Inject the test server client directly
    ring.server_client.client = llmring_server_client

    # Make a real chat request
    request = LLMRequest(
        model="anthropic:claude-3-5-haiku-20241022",
        messages=[Message(role="user", content="Say 'test' and nothing else.")],
        max_tokens=10,
    )

    response = await ring.chat(request)

    # Verify we got a real response
    assert response.content is not None
    assert len(response.content) > 0

    # Query the server to verify the log was stored
    logs_response = await llmring_server_client.get(
        "/api/v1/logs",
        params={"origin": "test-anthropic-logging", "limit": 1},
        headers=project_headers,
    )
    assert logs_response.status_code == 200

    logs = logs_response.json()
    assert len(logs) > 0

    log_entry = logs[0]
    assert log_entry["provider"] == "anthropic"
    assert "claude" in log_entry["model"].lower()
    assert log_entry["input_tokens"] > 0
    assert log_entry["output_tokens"] > 0
