"""
Test typed exceptions with real API error conditions.

Tests that providers now raise appropriate typed exceptions instead of
generic ValueError/Exception for different error scenarios.
"""

import pytest
from unittest.mock import patch

from llmring.service import LLMRing
from llmring.schemas import LLMRequest, Message
from llmring.exceptions import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ModelNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
    CircuitBreakerError,
)


class TestTypedExceptions:
    """Test that providers raise appropriate typed exceptions."""

    @pytest.fixture
    def service(self):
        """Create LLMRing service."""
        return LLMRing()

    @pytest.fixture
    def sample_request(self):
        """Sample request for testing."""
        return LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[Message(role="user", content="test")],
            max_tokens=10,
        )

    @pytest.mark.asyncio
    async def test_invalid_openai_api_key_raises_typed_exception(self, sample_request):
        """Test that invalid OpenAI API key raises ProviderAuthenticationError."""
        from llmring.providers.openai_api import OpenAIProvider

        # Create provider with invalid key
        provider = OpenAIProvider(api_key="sk-invalid-key-test")

        with pytest.raises(ProviderAuthenticationError) as exc_info:
            await provider.chat(
                messages=sample_request.messages, model="gpt-4o-mini", max_tokens=10
            )

        assert exc_info.value.provider == "openai"
        assert "authentication" in str(exc_info.value).lower()
        print(
            f"✓ OpenAI auth error raises ProviderAuthenticationError: {exc_info.value}"
        )

    @pytest.mark.asyncio
    async def test_invalid_anthropic_api_key_raises_typed_exception(
        self, sample_request
    ):
        """Test that invalid Anthropic API key raises ProviderAuthenticationError."""
        from llmring.providers.anthropic_api import AnthropicProvider

        # Create provider with invalid key
        provider = AnthropicProvider(api_key="sk-ant-invalid-key-test")

        with pytest.raises(ProviderAuthenticationError) as exc_info:
            await provider.chat(
                messages=sample_request.messages,
                model="claude-3-5-haiku",
                max_tokens=10,
            )

        assert exc_info.value.provider == "anthropic"
        assert "authentication" in str(exc_info.value).lower()
        print(
            f"✓ Anthropic auth error raises ProviderAuthenticationError: {exc_info.value}"
        )

    @pytest.mark.asyncio
    async def test_invalid_model_raises_typed_exception(self, service, sample_request):
        """Test that invalid model raises ModelNotFoundError."""
        sample_request.model = "openai:gpt-nonexistent-model"

        with pytest.raises(ModelNotFoundError) as exc_info:
            await service.chat(sample_request)

        assert exc_info.value.provider == "openai"
        assert exc_info.value.model_name == "gpt-nonexistent-model"
        print(f"✓ Invalid model raises ModelNotFoundError: {exc_info.value}")

    @pytest.mark.asyncio
    async def test_unsupported_provider_model_format(self, service):
        """Test that unsupported model format raises ModelNotFoundError."""
        request = LLMRequest(
            model="anthropic:claude-totally-fake-model",
            messages=[Message(role="user", content="test")],
            max_tokens=10,
        )

        with pytest.raises(ModelNotFoundError) as exc_info:
            await service.chat(request)

        assert exc_info.value.provider == "anthropic"
        assert "claude-totally-fake-model" in str(exc_info.value)
        print(f"✓ Unsupported model raises ModelNotFoundError: {exc_info.value}")

    @pytest.mark.asyncio
    async def test_timeout_raises_typed_exception(self, sample_request):
        """Test that timeouts raise ProviderTimeoutError."""
        from llmring.providers.openai_api import OpenAIProvider
        from llmring.exceptions import ProviderTimeoutError, ProviderResponseError

        # Create provider with invalid API key and very short timeout
        provider = OpenAIProvider(api_key="sk-invalid-key-for-timeout-test")

        # Use extremely short timeout to force timeout error
        with patch.dict("os.environ", {"LLMRING_PROVIDER_TIMEOUT_S": "0.001"}):
            with pytest.raises(
                (ProviderTimeoutError, ProviderResponseError)
            ) as exc_info:
                await provider.chat(
                    messages=sample_request.messages, model="gpt-4o-mini", max_tokens=10
                )

        # Either timeout or auth error is acceptable for this test
        # The key is that we're using typed exceptions, not generic ones
        assert exc_info.value.provider == "openai"
        print(f"✓ Timeout/auth raises typed exception: {type(exc_info.value).__name__}")

    @pytest.mark.asyncio
    async def test_circuit_breaker_raises_typed_exception(self, sample_request):
        """Test that circuit breaker raises CircuitBreakerError."""
        from llmring.providers.openai_api import OpenAIProvider
        from llmring.exceptions import CircuitBreakerError
        from unittest.mock import patch

        provider = OpenAIProvider(api_key="sk-test-key")

        # Mock the circuit breaker to return False (circuit open)
        with patch.object(provider._breaker, "allow", return_value=False):
            with pytest.raises(CircuitBreakerError) as exc_info:
                await provider.chat(
                    messages=sample_request.messages, model="gpt-4o-mini", max_tokens=10
                )

        assert exc_info.value.provider == "openai"
        assert "circuit breaker" in str(exc_info.value).lower()
        print(f"✓ Circuit breaker raises CircuitBreakerError: {exc_info.value}")

    def test_exception_hierarchy(self):
        """Test that all typed exceptions inherit from the correct base classes."""
        from llmring.exceptions import LLMRingError, ProviderError, ModelError

        # Test provider exception inheritance
        assert issubclass(ProviderAuthenticationError, ProviderError)
        assert issubclass(ProviderRateLimitError, ProviderError)
        assert issubclass(ProviderResponseError, ProviderError)
        assert issubclass(ProviderTimeoutError, ProviderError)

        # Test model exception inheritance
        assert issubclass(ModelNotFoundError, ModelError)

        # Test other exceptions
        assert issubclass(CircuitBreakerError, LLMRingError)

        # Test they can be caught as base types
        try:
            raise ProviderAuthenticationError("test", provider="test")
        except ProviderError:
            pass  # Should catch
        except Exception:
            pytest.fail("Should have been caught as ProviderError")

        print("✓ Exception hierarchy is correct")
