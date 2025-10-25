from types import SimpleNamespace

import pytest

from llmring.cli import cmd_chat
from llmring.schemas import LLMResponse, StreamChunk


class DummyRing:
    """Minimal stand-in for LLMRing used to exercise CLI behaviour."""

    def __init__(self):
        self.chat_called = False
        self.chat_stream_called = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def resolve_alias(self, alias, profile=None):
        # Only invoked for alias-based requests; our test uses provider:model.
        return alias

    def chat_stream(self, request, profile=None):
        self.chat_stream_called = True

        async def _generator():
            yield StreamChunk(delta="Hello", model="openai:gpt-4")
            yield StreamChunk(
                delta="",
                model="openai:gpt-4",
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

        return _generator()

    async def chat(self, request, profile=None):
        self.chat_called = True
        return LLMResponse(content="fallback", model="openai:gpt-4")


@pytest.mark.asyncio
async def test_cmd_chat_stream_uses_chat_stream(monkeypatch, capsys):
    dummy_ring = DummyRing()

    # Patch LLMRing constructor to return our dummy context manager.
    monkeypatch.setattr("llmring.cli.LLMRing", lambda *args, **kwargs: dummy_ring)

    args = SimpleNamespace(
        model="openai:gpt-4",
        message="Hi there",
        system=None,
        temperature=None,
        max_tokens=None,
        json=False,
        verbose=False,
        stream=True,
        profile=None,
    )

    result = await cmd_chat(args)

    captured = capsys.readouterr()

    assert result == 0
    assert dummy_ring.chat_stream_called is True
    assert dummy_ring.chat_called is False
    # Combined stream output should contain the streamed text.
    assert "Hello" in captured.out
