"""Tests for CLI server helper functions."""

from pathlib import Path

import pytest

from llmring.cli import _load_env_file, _resolve_server_settings, _write_env_file
from llmring.service import DEFAULT_SAAS_URL


def test_resolve_server_prefers_overrides(monkeypatch):
    """Explicit CLI overrides should take precedence over env values."""
    monkeypatch.setenv("LLMRING_SERVER_URL", "http://env-server:8000")
    monkeypatch.setenv("LLMRING_API_KEY", "env-key")
    server, api_key, used = _resolve_server_settings(
        server_override="https://override", api_key_override="override-key"
    )
    assert server == "https://override"
    assert api_key == "override-key"
    assert used is False


def test_resolve_server_env(monkeypatch):
    """Environment variables should be used when overrides are absent."""
    monkeypatch.setenv("LLMRING_SERVER_URL", "http://env-server:8000/")
    monkeypatch.setenv("LLMRING_API_KEY", "env-key")
    monkeypatch.delenv("LLMRING_PREFER_SAAS", raising=False)

    server, api_key, used = _resolve_server_settings()
    assert server == "http://env-server:8000"
    assert api_key == "env-key"
    assert used is False


def test_resolve_server_saas_fallback(monkeypatch):
    """LLMRING_PREFER_SAAS enables fallback to hosted API."""
    monkeypatch.delenv("LLMRING_SERVER_URL", raising=False)
    monkeypatch.delenv("LLMRING_API_KEY", raising=False)
    monkeypatch.setenv("LLMRING_PREFER_SAAS", "true")

    server, api_key, used = _resolve_server_settings()
    assert server == DEFAULT_SAAS_URL
    assert api_key is None
    assert used is True


def test_disable_saas_fallback(monkeypatch):
    """Callers can opt out of SaaS fallback."""
    monkeypatch.delenv("LLMRING_SERVER_URL", raising=False)
    monkeypatch.delenv("LLMRING_API_KEY", raising=False)
    monkeypatch.setenv("LLMRING_PREFER_SAAS", "1")

    server, api_key, used = _resolve_server_settings(enable_saas_fallback=False)
    assert server is None
    assert api_key is None
    assert used is False


def test_write_and_load_env_file(tmp_path: Path):
    """Writing an env file should be idempotent and reloadable."""
    env_file = tmp_path / ".env.llmring"
    _write_env_file(env_file, "http://localhost:8000", "llmr_pk_123", overwrite=False)

    contents = env_file.read_text(encoding="utf-8")
    assert "export LLMRING_SERVER_URL=http://localhost:8000" in contents
    assert "export LLMRING_API_KEY=llmr_pk_123" in contents

    data = _load_env_file(env_file)
    assert data["LLMRING_SERVER_URL"] == "http://localhost:8000"
    assert data["LLMRING_API_KEY"] == "llmr_pk_123"

    # Overwrite should succeed
    _write_env_file(env_file, "http://localhost:9000", "llmr_pk_456", overwrite=True)
    contents = env_file.read_text(encoding="utf-8")
    assert "export LLMRING_SERVER_URL=http://localhost:9000" in contents
    assert "export LLMRING_API_KEY=llmr_pk_456" in contents
    data = _load_env_file(env_file)
    assert data["LLMRING_SERVER_URL"] == "http://localhost:9000"
    assert data["LLMRING_API_KEY"] == "llmr_pk_456"
