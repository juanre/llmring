"""Tests for runtime fallback across alias model pools."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from llmring import LLMRing
from llmring.base import BaseLLMProvider, ProviderConfig
from llmring.exceptions import ProviderAuthenticationError, ProviderTimeoutError
from llmring.lockfile_core import Lockfile
from llmring.schemas import FileMetadata, LLMRequest, LLMResponse, Message, StreamChunk


class FakeProvider(BaseLLMProvider):
    """Minimal provider used to validate alias fallback behavior without mocks."""

    def __init__(self) -> None:
        super().__init__(
            ProviderConfig(api_key=None, base_url=None, default_model=None, timeout_seconds=None)
        )
        self.chat_calls: list[str] = []
        self.stream_calls: list[str] = []

    async def chat(self, messages, model: str, **kwargs: Any) -> LLMResponse:  # type: ignore[override]
        self.chat_calls.append(model)
        if model == "fail-model":
            raise ProviderTimeoutError("timed out", provider="ollama")
        if model == "auth-fail-model":
            raise ProviderAuthenticationError("bad key", provider="ollama")
        return LLMResponse(content="ok", model=model, finish_reason="stop")

    async def chat_stream(  # type: ignore[override]
        self, messages, model: str, **kwargs: Any
    ) -> AsyncIterator[StreamChunk]:
        self.stream_calls.append(model)
        if model == "fail-model":
            raise ProviderTimeoutError("timed out", provider="ollama")

        async def _gen() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(delta="stream-ok", model=model)
            if model == "fail-mid-stream-model":
                raise ProviderTimeoutError("mid-stream failure", provider="ollama")
            yield StreamChunk(delta="", model=model, finish_reason="stop")

        return _gen()

    async def get_capabilities(self):  # type: ignore[override]
        return {}

    async def get_default_model(self) -> str:  # type: ignore[override]
        return "ok-model"

    async def upload_file(
        self, file, purpose: str = "analysis", filename: str | None = None, **kwargs
    ):
        raise NotImplementedError

    async def delete_file(self, file_id: str) -> bool:
        raise NotImplementedError

    async def list_files(self, purpose: str | None = None, limit: int = 100) -> list[FileMetadata]:
        raise NotImplementedError

    async def get_file(self, file_id: str) -> FileMetadata:
        raise NotImplementedError


def _write_minimal_registry(tmp_path: Path) -> str:
    registry_root = tmp_path / "registry"
    (registry_root / "ollama").mkdir(parents=True)
    (registry_root / "ollama" / "models.json").write_text(json.dumps({"models": {}}))
    return f"file://{registry_root}"


def _write_lockfile(tmp_path: Path, models: list[str]) -> Path:
    lockfile = Lockfile.create_default()
    lockfile.set_binding("demo", models, profile="default")
    lockfile_path = tmp_path / "llmring.lock.json"
    lockfile.save(lockfile_path)
    return lockfile_path


@pytest.mark.asyncio
async def test_chat_falls_back_to_next_model_on_provider_error(tmp_path: Path) -> None:
    registry_url = _write_minimal_registry(tmp_path)
    lockfile_path = _write_lockfile(tmp_path, ["ollama:fail-model", "ollama:ok-model"])

    ring = LLMRing(registry_url=registry_url, lockfile_path=str(lockfile_path))
    fake = FakeProvider()
    ring.providers["ollama"] = fake

    request = LLMRequest(model="demo", messages=[Message(role="user", content="hi")])
    response = await ring.chat(request)

    assert response.content == "ok"
    assert response.model == "ollama:ok-model"
    assert fake.chat_calls == ["fail-model", "ok-model"]


@pytest.mark.asyncio
async def test_chat_stream_falls_back_before_first_chunk(tmp_path: Path) -> None:
    registry_url = _write_minimal_registry(tmp_path)
    lockfile_path = _write_lockfile(tmp_path, ["ollama:fail-model", "ollama:ok-model"])

    ring = LLMRing(registry_url=registry_url, lockfile_path=str(lockfile_path))
    fake = FakeProvider()
    ring.providers["ollama"] = fake

    request = LLMRequest(model="demo", messages=[Message(role="user", content="hi")])
    chunks: list[StreamChunk] = []
    async for chunk in ring.chat_stream(request):
        chunks.append(chunk)

    assert "".join(chunk.delta for chunk in chunks) == "stream-ok"
    assert fake.stream_calls == ["fail-model", "ok-model"]


@pytest.mark.asyncio
async def test_chat_does_not_fallback_on_auth_error(tmp_path: Path) -> None:
    registry_url = _write_minimal_registry(tmp_path)
    lockfile_path = _write_lockfile(tmp_path, ["ollama:auth-fail-model", "ollama:ok-model"])

    ring = LLMRing(registry_url=registry_url, lockfile_path=str(lockfile_path))
    ring.providers["ollama"] = FakeProvider()

    request = LLMRequest(model="demo", messages=[Message(role="user", content="hi")])
    with pytest.raises(ProviderAuthenticationError):
        await ring.chat(request)


@pytest.mark.asyncio
async def test_chat_stream_does_not_fallback_after_first_chunk(tmp_path: Path) -> None:
    """After yielding the first chunk, mid-stream failures must NOT trigger fallback.

    Once streaming has started, the client has already received partial data.
    Falling back to another model would cause duplicate or inconsistent output.
    """
    registry_url = _write_minimal_registry(tmp_path)
    lockfile_path = _write_lockfile(tmp_path, ["ollama:fail-mid-stream-model", "ollama:ok-model"])

    ring = LLMRing(registry_url=registry_url, lockfile_path=str(lockfile_path))
    fake = FakeProvider()
    ring.providers["ollama"] = fake

    request = LLMRequest(model="demo", messages=[Message(role="user", content="hi")])
    chunks: list[StreamChunk] = []

    with pytest.raises(ProviderTimeoutError, match="mid-stream failure"):
        async for chunk in ring.chat_stream(request):
            chunks.append(chunk)

    # First chunk was received before the failure
    assert len(chunks) == 1
    assert chunks[0].delta == "stream-ok"

    # No fallback was attempted - only the primary model was called
    assert fake.stream_calls == ["fail-mid-stream-model"]
