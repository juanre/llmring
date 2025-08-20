import os

import pytest

from llmring.net.safe_fetcher import SafeFetchError, fetch_bytes, get_default_config


@pytest.mark.asyncio
async def test_fetch_disallowed_by_default():
    # Ensure default denies
    os.environ.pop("LLMRING_ALLOW_REMOTE_URLS", None)
    cfg = get_default_config()
    assert cfg["allow_remote_urls"] is False
    with pytest.raises(SafeFetchError):
        await fetch_bytes("https://example.com/image.png", cfg)


@pytest.mark.asyncio
async def test_fetch_https_only_and_allowlist(monkeypatch):
    # Enable but restrict to allowlist
    monkeypatch.setenv("LLMRING_ALLOW_REMOTE_URLS", "true")
    monkeypatch.setenv("LLMRING_ALLOWED_HOSTS", "allowed.example")
    cfg = get_default_config()
    assert cfg["allow_remote_urls"] is True
    # Wrong host rejected
    with pytest.raises(SafeFetchError):
        await fetch_bytes("https://not-allowed.example/image.png", cfg)
    # HTTP rejected
    with pytest.raises(SafeFetchError):
        await fetch_bytes("http://allowed.example/image.png", cfg)


@pytest.mark.asyncio
async def test_fetch_size_and_type_limits(monkeypatch):
    # This is a behavioral test; we can't hit the network in CI, so we just verify config plumbing
    monkeypatch.setenv("LLMRING_ALLOW_REMOTE_URLS", "true")
    monkeypatch.setenv("LLMRING_ALLOWED_HOSTS", "example.com")
    monkeypatch.setenv("LLMRING_MAX_DOWNLOAD_SIZE_BYTES", "1024")
    monkeypatch.setenv("LLMRING_ALLOWED_CONTENT_TYPES", "image/png")
    cfg = get_default_config()
    # We can't actually fetch, but we can at least assert cfg picks up values
    assert cfg["max_size_bytes"] == 1024
    assert cfg["content_types_allowed"] == ["image/png"]
