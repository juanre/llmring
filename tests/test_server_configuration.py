"""Tests for server auto-configuration behaviour."""

import asyncio

import pytest

from llmring.service import DEFAULT_SAAS_URL, LLMRing


@pytest.mark.asyncio
async def test_env_server_configured_by_default(monkeypatch):
    """LLMRing should pick up server configuration from environment variables."""
    monkeypatch.setenv("LLMRING_SERVER_URL", "http://localhost:9000")
    monkeypatch.setenv("LLMRING_API_KEY", "test-key")
    monkeypatch.delenv("LLMRING_PREFER_SAAS", raising=False)

    ring = LLMRing()

    try:
        assert ring.server_client is not None
        assert ring.server_url == "http://localhost:9000"
        assert ring.server_client.base_url == "http://localhost:9000"
    finally:
        if ring.server_client:
            await ring.server_client.close()


def test_no_server_when_unconfigured(monkeypatch):
    """Without env variables or args, LLMRing should not create a server client."""
    monkeypatch.delenv("LLMRING_SERVER_URL", raising=False)
    monkeypatch.delenv("LLMRING_API_KEY", raising=False)
    monkeypatch.delenv("LLMRING_PREFER_SAAS", raising=False)

    ring = LLMRing()
    assert ring.server_client is None
    assert ring.server_url is None


@pytest.mark.asyncio
async def test_prefer_saas_opt_in(monkeypatch):
    """SaaS fallback should require explicit opt-in via LLMRING_PREFER_SAAS."""
    monkeypatch.delenv("LLMRING_SERVER_URL", raising=False)
    monkeypatch.delenv("LLMRING_API_KEY", raising=False)
    monkeypatch.setenv("LLMRING_PREFER_SAAS", "1")

    ring = LLMRing()

    try:
        assert ring.server_client is not None
        assert ring.server_client.base_url == DEFAULT_SAAS_URL
    finally:
        if ring.server_client:
            await ring.server_client.close()
