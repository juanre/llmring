"""
Unit tests for the base LLM provider class.
"""

import pytest
from llmring.base import BaseLLMProvider
from llmring.schemas import LLMResponse


@pytest.mark.llm
@pytest.mark.unit
class TestBaseLLMProvider:
    """Test the abstract base LLM provider class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that the abstract base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMProvider()

    def test_abstract_methods_must_be_implemented(self):
        """Test that concrete implementations must implement all abstract methods."""

        class IncompleteProvider(BaseLLMProvider):
            """Incomplete provider missing required methods."""

            pass

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_complete_implementation_works(self):
        """Test that a complete implementation can be instantiated."""

        class CompleteProvider(BaseLLMProvider):
            """Complete provider implementation for testing."""

            def __init__(self, api_key=None, base_url=None):
                super().__init__(api_key, base_url)

            async def chat(self, messages, model, **kwargs):
                return LLMResponse(
                    content="Test response",
                    model=model,
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                )

            def get_token_count(self, text):
                return len(text.split())

            def validate_model(self, model):
                return model == "test-model"

            def get_supported_models(self):
                return ["test-model"]

            def get_default_model(self):
                return "test-model"

        provider = CompleteProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.base_url is None

    def test_get_default_model_fallback(self):
        """Test that get_default_model falls back to first supported model."""

        class TestProvider(BaseLLMProvider):
            """Test provider for default model testing."""

            def __init__(self):
                super().__init__(api_key="test")

            async def chat(self, messages, model, **kwargs):
                return LLMResponse(content="test", model=model)

            def get_token_count(self, text):
                return len(text.split())

            def validate_model(self, model):
                return True

            def get_supported_models(self):
                return ["model-1", "model-2", "model-3"]

            def get_default_model(self):
                return "model-1"

        provider = TestProvider()
        assert provider.get_default_model() == "model-1"

    def test_get_default_model_empty_list(self):
        """Test get_default_model when no models are supported."""

        class EmptyProvider(BaseLLMProvider):
            """Provider with no supported models."""

            def __init__(self):
                super().__init__(api_key="test")

            async def chat(self, messages, model, **kwargs):
                return LLMResponse(content="test", model=model)

            def get_token_count(self, text):
                return len(text.split())

            def validate_model(self, model):
                return False

            def get_supported_models(self):
                return []

            def get_default_model(self):
                return "default"

        provider = EmptyProvider()
        assert provider.get_default_model() == "default"
