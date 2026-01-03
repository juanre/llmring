from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Dict, List, Optional

import pytest

from llmring.base import (
    TIMEOUT_UNSET,
    BaseLLMProvider,
    ProviderCapabilities,
    ProviderConfig,
    TimeoutSetting,
)
from llmring.schemas import FileUploadResponse, LLMRequest, LLMResponse, Message, StreamChunk
from llmring.service import LLMRing


class DummyRing(LLMRing):
    """LLMRing subclass that suppresses auto provider registration for tests."""

    def __init__(self, *args, **kwargs):
        self._suppress_auto_providers = True
        super().__init__(*args, **kwargs)
        self._suppress_auto_providers = False

    def register_provider(self, provider_type: str, **kwargs):
        if getattr(self, "_suppress_auto_providers", False):
            return
        super().register_provider(provider_type, **kwargs)


class DummyProvider(BaseLLMProvider):
    """Provider that records the last timeout it received."""

    def __init__(self):
        super().__init__(ProviderConfig())
        self.last_timeout: Optional[float] = None

    async def chat(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        json_response: Optional[bool] = None,
        cache: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        files: Optional[List[str]] = None,
        timeout: TimeoutSetting = TIMEOUT_UNSET,
    ) -> LLMResponse:
        self.last_timeout = self._resolve_timeout_value(timeout)
        return LLMResponse(content="ok", model=model)

    async def chat_stream(
        self,
        messages: List[Message],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        json_response: Optional[bool] = None,
        cache: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        files: Optional[List[str]] = None,
        timeout: TimeoutSetting = TIMEOUT_UNSET,
    ) -> AsyncIterator[StreamChunk]:
        self.last_timeout = self._resolve_timeout_value(timeout)
        yield StreamChunk(delta="", model=model, finish_reason="stop")

    async def get_default_model(self) -> str:
        return "dummy-model"

    async def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_name="dummy",
            supported_models=["dummy-model"],
            supports_streaming=True,
            supports_tools=False,
            supports_vision=False,
            supports_audio=False,
            supports_documents=False,
            supports_json_mode=False,
            supports_caching=False,
            max_context_window=0,
            default_model="dummy-model",
        )

    async def upload_file(
        self,
        file: str | Path | BinaryIO,
        purpose: str = "analysis",
        filename: str | None = None,
        **kwargs: Any,
    ) -> FileUploadResponse:
        raise NotImplementedError

    async def delete_file(self, file_id: str) -> bool:
        raise NotImplementedError

    async def list_files(self, purpose: str | None = None, limit: int = 100) -> list[Any]:
        raise NotImplementedError

    async def get_file(self, file_id: str) -> Any:
        raise NotImplementedError


def _prepare_ring(timeout: Optional[float]) -> tuple[DummyRing, DummyProvider]:
    ring = DummyRing(timeout=timeout)
    provider = DummyProvider()
    ring.providers = {"dummy": provider}
    if ring._alias_resolver:
        ring._alias_resolver.update_available_providers({"dummy"})
    return ring, provider


def _basic_request(**kwargs) -> LLMRequest:
    params = {
        "model": "dummy:dummy-model",
        "messages": [Message(role="user", content="ping")],
    }
    params.update(kwargs)
    return LLMRequest(**params)


@pytest.mark.asyncio
async def test_ring_uses_default_timeout(monkeypatch):
    monkeypatch.delenv("LLMRING_PROVIDER_TIMEOUT_S", raising=False)
    ring, provider = _prepare_ring(timeout=123.0)

    await ring.chat(_basic_request())

    assert provider.last_timeout == 123.0


@pytest.mark.asyncio
async def test_request_override_timeout():
    ring, provider = _prepare_ring(timeout=50.0)
    await ring.chat(_basic_request(timeout=7.5))
    assert provider.last_timeout == 7.5


@pytest.mark.asyncio
async def test_request_can_disable_timeout():
    ring, provider = _prepare_ring(timeout=12.0)
    await ring.chat(_basic_request(timeout=None))
    assert provider.last_timeout is None


@pytest.mark.asyncio
async def test_env_timeout_used_when_not_specified(monkeypatch):
    monkeypatch.setenv("LLMRING_PROVIDER_TIMEOUT_S", "9.25")
    ring = DummyRing()
    provider = DummyProvider()
    ring.providers = {"dummy": provider}
    if ring._alias_resolver:
        ring._alias_resolver.update_available_providers({"dummy"})

    await ring.chat(_basic_request())
    assert provider.last_timeout == pytest.approx(9.25)
