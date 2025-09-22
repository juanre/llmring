"""
Integration tests for the LLM service with real providers.
"""

import asyncio
import os

import pytest

from llmring.exceptions import ProviderNotFoundError
from llmring.schemas import LLMRequest, LLMResponse, Message


@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.slow
class TestLLMRingIntegration:
    """Integration tests for LLMRing with real providers."""

    # llmring fixture is provided by conftest.py

    @pytest.mark.asyncio
    async def test_anthropic_provider_integration(self, llmring):
        """Test Anthropic provider through LLM service."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not found in environment")

        # Register Anthropic provider
        llmring.register_provider("anthropic", api_key=api_key)

        # Create request
        request = LLMRequest(
            messages=[Message(role="user", content="Say 'Hello from Anthropic'")],
            model="mvp_test",
            max_tokens=20,
        )

        # Send request
        response = await llmring.chat(request)

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert "hello" in response.content.lower()
        assert "opus" in response.model.lower()  # Should be Opus 4.1 model

    @pytest.mark.asyncio
    async def test_openai_provider_integration(self, llmring):
        """Test OpenAI provider through LLM service."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not found in environment")

        # Register OpenAI provider
        llmring.register_provider("openai", api_key=api_key)

        # Create request
        request = LLMRequest(
            messages=[Message(role="user", content="Say 'Hello from OpenAI'")],
            model="openai_fast",
            max_tokens=20,
        )

        # Send request
        response = await llmring.chat(request)

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert "hello" in response.content.lower()
        assert "turbo" in response.model or "mini" in response.model

    @pytest.mark.asyncio
    @pytest.mark.google
    async def test_google_provider_integration(self, llmring):
        """Test Google provider through LLM service."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment")

        # Skip if we've been hitting rate limits
        if os.getenv("SKIP_GOOGLE_TESTS"):
            pytest.skip("Skipping Google tests due to rate limits")

        try:
            # Register Google provider
            llmring.register_provider("google", api_key=api_key)

            # Create request
            request = LLMRequest(
                messages=[Message(role="user", content="Say 'Hello from Google'")],
                model="long_context",
                max_tokens=20,
            )

            # Send request
            response = await llmring.chat(request)

            assert isinstance(response, LLMResponse)
            assert response.content is not None
            assert "hello" in response.content.lower()
            assert "gemini" in response.model or "turbo" in response.model
        except Exception as e:
            error_msg = str(e).lower()
            # Check for quota/rate limit errors
            if any(
                term in error_msg
                for term in [
                    "quota",
                    "rate limit",
                    "resource_exhausted",
                    "429",
                    "billing",
                    "exceeded",
                ]
            ):
                pytest.skip(f"Google API quota exceeded: {str(e)[:100]}")
            raise

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Ollama tests take too long - skipping for faster test runs")
    @pytest.mark.ollama
    async def test_ollama_provider_integration(self, llmring):
        """Test Ollama provider through LLM service."""
        # Check if Ollama is available
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code != 200:
                    pytest.skip("Ollama not running at localhost:11434")
        except (httpx.ConnectError, httpx.RequestError):
            pytest.skip("Ollama not running at localhost:11434")

        # Register Ollama provider
        llmring.register_provider("ollama")

        # Create request
        request = LLMRequest(
            messages=[Message(role="user", content="Say 'Hello from Ollama'")],
            model="local",
            max_tokens=20,
        )

        # Send request
        response = await llmring.chat(request)

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert "llama" in response.model

    @pytest.mark.asyncio
    async def test_provider_switching(self, llmring):
        """Test switching between providers."""
        # Try to get at least one API key
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if not anthropic_key and not openai_key:
            pytest.skip("Need at least one API key for testing")

        providers_tested = 0

        if anthropic_key:
            llmring.register_provider("anthropic", api_key=anthropic_key)
            request = LLMRequest(
                messages=[Message(role="user", content="Say 'test'")],
                model="anthropic_balanced",
                max_tokens=10,
            )
            response = await llmring.chat(request)
            assert isinstance(response, LLMResponse)
            providers_tested += 1

        if openai_key:
            llmring.register_provider("openai", api_key=openai_key)
            request = LLMRequest(
                messages=[Message(role="user", content="Say 'test'")],
                model="openai_fast",
                max_tokens=10,
            )
            response = await llmring.chat(request)
            assert isinstance(response, LLMResponse)
            providers_tested += 1

        assert providers_tested >= 1, "At least one provider should have been tested"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, llmring):
        """Test handling concurrent requests to same provider."""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No API key found for testing")

        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
            model = "fast"
            llmring.register_provider(provider, api_key=os.getenv("OPENAI_API_KEY"))
        else:
            provider = "anthropic"
            model = "balanced"
            llmring.register_provider(provider, api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Create multiple requests
        requests = []
        for i in range(3):
            request = LLMRequest(
                messages=[Message(role="user", content=f"Say exactly 'Response {i}'")],
                model=model,
                max_tokens=10,
            )
            requests.append(llmring.chat(request))

        # Execute concurrently
        responses = await asyncio.gather(*requests)

        # Verify all responses
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert isinstance(response, LLMResponse)
            assert response.content is not None
            assert f"{i}" in response.content or "Response" in response.content

    @pytest.mark.asyncio
    async def test_error_handling_invalid_provider(self, llmring):
        """Test error handling for invalid provider."""
        request = LLMRequest(messages=[Message(role="user", content="test")], model="invalid:model")

        with pytest.raises(ProviderNotFoundError, match="Provider .* not found"):
            await llmring.chat(request)

    @pytest.mark.asyncio
    async def test_error_handling_missing_api_key(self, llmring):
        """Test error handling for missing API key."""
        # Use a provider that's definitely not registered
        request = LLMRequest(
            messages=[Message(role="user", content="test")],
            model="definitely_not_a_provider:test",
        )

        with pytest.raises(ProviderNotFoundError, match="Provider .* not found"):
            await llmring.chat(request)
