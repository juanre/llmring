"""
Test that all providers properly implement the base interface.

This ensures the base class signature matches provider implementations,
fixing the LSP violation identified in the code review.
"""

import pytest

from llmring.base import BaseLLMProvider
from llmring.providers.anthropic_api import AnthropicProvider
from llmring.providers.google_api import GoogleProvider
from llmring.providers.ollama_api import OllamaProvider
from llmring.providers.openai_api import OpenAIProvider
from llmring.schemas import LLMRequest, Message


class TestBaseInterfaceCompliance:
    """Test that providers conform to the base interface."""

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing."""
        return [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello!"),
        ]

    @pytest.fixture
    def sample_request(self, sample_messages):
        """Sample LLMRequest for testing."""
        return LLMRequest(
            model="test-model",
            messages=sample_messages,
            temperature=0.7,
            max_tokens=100,
            extra_params={"test_param": "test_value"},
        )

    def test_base_interface_signature(self):
        """Test that base class has the correct signature."""
        import inspect

        sig = inspect.signature(BaseLLMProvider.chat)
        params = list(sig.parameters.keys())

        expected_params = [
            "self",
            "messages",
            "model",
            "temperature",
            "max_tokens",
            "reasoning_tokens",
            "response_format",
            "tools",
            "tool_choice",
            "json_response",
            "cache",
            "extra_params",
            "files",
        ]

        assert (
            params == expected_params
        ), f"Base interface signature mismatch. Expected {expected_params}, got {params}"

    @pytest.mark.parametrize(
        "provider_class",
        [OpenAIProvider, AnthropicProvider, GoogleProvider, OllamaProvider],
    )
    def test_provider_signature_matches_base(self, provider_class):
        """Test that each provider's chat method matches the base signature."""
        import inspect

        base_sig = inspect.signature(BaseLLMProvider.chat)
        provider_sig = inspect.signature(provider_class.chat)

        # Get parameter names (excluding 'self')
        base_params = list(base_sig.parameters.keys())[1:]
        provider_params = list(provider_sig.parameters.keys())[1:]

        assert provider_params == base_params, (
            f"{provider_class.__name__}.chat() signature doesn't match base. "
            f"Expected {base_params}, got {provider_params}"
        )

    # Note: Provider functionality is validated via real API tests in integration/
    # These mock-based tests were removed to avoid complex mock setup issues
    # Real API validation is in:
    # - tests/integration/test_service_extended_fixes.py
    # - tests/integration/test_extra_params_simple.py
    # - tests/integration/test_provider_enhancements.py

    def test_extra_params_parameter_present(self):
        """Test that all providers accept extra_params parameter."""
        import inspect

        providers = [OpenAIProvider, AnthropicProvider, GoogleProvider, OllamaProvider]

        for provider_class in providers:
            sig = inspect.signature(provider_class.chat)
            assert (
                "extra_params" in sig.parameters
            ), f"{provider_class.__name__}.chat() missing extra_params parameter"

            # Check it has correct type annotation
            param = sig.parameters["extra_params"]
            assert (
                param.default is None
            ), f"{provider_class.__name__} extra_params should default to None"
