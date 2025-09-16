"""Test that models work when registry is unavailable."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
from llmring import LLMRing
from llmring.schemas import LLMRequest
from llmring.registry import RegistryClient


@pytest.mark.asyncio
async def test_model_validation_with_unavailable_registry():
    """Test that models validate successfully when registry is unavailable."""

    # Create LLMRing with an invalid registry URL
    # Use a cache dir that doesn't exist to avoid cached data
    with patch.object(RegistryClient, 'CACHE_DIR', Path("/tmp/test-no-cache")):
        ring = LLMRing(registry_url="http://invalid-registry-url.example.com")

        # Get provider
        provider = ring.providers.get("openai")
        if provider:
            # Test that validation returns True when registry is unavailable
            is_valid = await provider.validate_model("gpt-4o-mini")
            assert is_valid, "Model should validate as True when registry is unavailable"

            # Even completely fake models should validate as True
            is_valid = await provider.validate_model("completely-fake-model")
            assert is_valid, "Fake model should validate as True when registry is unavailable"


@pytest.mark.asyncio
async def test_registry_client_fallback():
    """Test that RegistryClient returns True when fetch fails."""

    # Use a non-existent cache directory to ensure no cached data
    with patch.object(RegistryClient, 'CACHE_DIR', Path("/tmp/test-no-cache")):
        registry = RegistryClient("http://invalid-url.example.com")

        # Clear any in-memory cache
        registry._cache.clear()

        # Test validate_model returns True on fetch failure
        is_valid = await registry.validate_model("openai", "any-model")
        assert is_valid, "Registry should return True when fetch fails"


@pytest.mark.asyncio
async def test_chat_works_without_registry():
    """Test that chat requests work when registry is unavailable."""

    # Use a non-existent cache directory
    with patch.object(RegistryClient, 'CACHE_DIR', Path("/tmp/test-no-cache")):
        # Create LLMRing with invalid registry
        ring = LLMRing(registry_url="http://invalid-registry-url.example.com")

        # Mock the OpenAI provider's chat method to avoid actual API calls
        provider = ring.providers.get("openai")
        if provider:
            with patch.object(provider, "chat", new_callable=AsyncMock) as mock_chat:
                from llmring.schemas import LLMResponse

                mock_chat.return_value = LLMResponse(
                    content="Hello!",
                    model="openai:gpt-4o-mini",
                    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    provider="openai"
                )

                # This should work even though gpt-4o-mini might not be in registry
                request = LLMRequest(
                    model="openai:gpt-4o-mini",
                    messages=[{"role": "user", "content": "Say hello"}]
                )

                response = await ring.chat(request)
                assert response.content == "Hello!"
                mock_chat.assert_called_once()


@pytest.mark.asyncio
async def test_providers_share_registry_client():
    """Test that providers share the same registry client as the service."""

    with patch.object(RegistryClient, 'CACHE_DIR', Path("/tmp/test-no-cache")):
        ring = LLMRing(registry_url="http://custom-registry.example.com")

        for provider_name in ["openai", "anthropic", "google", "ollama"]:
            provider = ring.providers.get(provider_name)
            if provider and hasattr(provider, '_registry_client'):
                assert provider._registry_client is ring.registry, \
                    f"{provider_name} provider should share the service's registry client"
                assert provider._registry_client.registry_url == "http://custom-registry.example.com", \
                    f"{provider_name} provider should use the custom registry URL"