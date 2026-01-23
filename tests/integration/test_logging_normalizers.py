"""
Integration tests for logging normalizers using real provider SDK responses.

These tests verify that the normalizers correctly extract data from actual
provider SDK response objects, not mocks.
"""

import os

import pytest

from llmring.logging.normalizers import detect_provider, normalize_response


class TestNormalizersWithRealResponses:
    """Test normalizers with actual SDK response objects from real API calls."""

    @pytest.mark.asyncio
    async def test_normalize_openai_response(self):
        """Test normalizer with real OpenAI SDK response."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
            max_tokens=10,
        )

        # Test provider detection
        provider = detect_provider(response)
        assert provider == "openai", f"Expected 'openai', got '{provider}'"

        # Test normalization
        content, model, usage, finish_reason = normalize_response(response, "openai")

        assert isinstance(content, str)
        assert len(content) > 0
        assert "gpt-4o-mini" in model
        assert isinstance(usage, dict)
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_normalize_anthropic_response(self):
        """Test normalizer with real Anthropic SDK response."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        import anthropic

        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
        )

        # Test provider detection
        provider = detect_provider(response)
        assert provider == "anthropic", f"Expected 'anthropic', got '{provider}'"

        # Test normalization
        content, model, usage, finish_reason = normalize_response(response, "anthropic")

        assert isinstance(content, str)
        assert len(content) > 0
        assert "claude" in model.lower()
        assert isinstance(usage, dict)
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert finish_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_normalize_google_response(self):
        """Test normalizer with real Google SDK response."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Google SDK uses sync API, wrap in asyncio
        import asyncio

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content("Say 'test' and nothing else."),
        )

        # Test provider detection
        provider = detect_provider(response)
        assert provider == "google", f"Expected 'google', got '{provider}'"

        # Test normalization
        content, model_name, usage, finish_reason = normalize_response(response, "google")

        assert isinstance(content, str)
        assert len(content) > 0
        assert isinstance(usage, dict)
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage


class TestProviderDetectionWithRealTypes:
    """Test provider detection with real SDK types."""

    def test_detect_openai_type_module(self):
        """Test that OpenAI response type module is detected correctly."""
        try:
            from openai.types.chat import ChatCompletion

            # Check the module name
            assert "openai" in ChatCompletion.__module__.lower()
        except ImportError:
            pytest.skip("openai package not installed")

    def test_detect_anthropic_type_module(self):
        """Test that Anthropic response type module is detected correctly."""
        try:
            from anthropic.types import Message

            # Check the module name
            assert "anthropic" in Message.__module__.lower()
        except ImportError:
            pytest.skip("anthropic package not installed")

    def test_detect_google_type_module(self):
        """Test that Google response type module is detected correctly."""
        try:
            from google.generativeai.types import GenerateContentResponse

            # Check the module name
            module = GenerateContentResponse.__module__.lower()
            assert "google" in module or "genai" in module
        except ImportError:
            pytest.skip("google-generativeai package not installed")
