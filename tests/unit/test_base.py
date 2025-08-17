"""
Unit tests for the base LLM provider class.
"""

import pytest
from llmring.base import BaseLLMProvider, LLMProviderFactory
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

    @pytest.mark.asyncio
    async def test_generate_response_convenience_method(self):
        """Test the convenience method for generating responses."""

        class TestProvider(BaseLLMProvider):
            """Test provider for convenience method testing."""

            def __init__(self):
                super().__init__(api_key="test")

            async def chat(self, messages, model, **kwargs):
                return LLMResponse(
                    content="Generated response",
                    model=model,
                    usage={"prompt_tokens": 15, "completion_tokens": 8},
                )

            def get_token_count(self, text):
                return len(text.split())

            def validate_model(self, model):
                return True

            def get_supported_models(self):
                return ["default-model"]

            def get_default_model(self):
                return "default-model"

        provider = TestProvider()
        response = await provider.generate_response(
            "You are a test assistant", "Say hello"
        )
        assert response == "Generated response"

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


@pytest.mark.llm
@pytest.mark.unit
class TestLLMProviderFactory:
    """Test the LLM provider factory."""

    def test_create_openai_provider(self):
        """Test creating an OpenAI provider."""
        provider = LLMProviderFactory.create_provider("openai", api_key="test-key")
        assert provider.__class__.__name__ == "OpenAIProvider"
        assert provider.api_key == "test-key"

    def test_create_anthropic_provider(self):
        """Test creating an Anthropic provider."""
        provider = LLMProviderFactory.create_provider("anthropic", api_key="test-key")
        assert provider.__class__.__name__ == "AnthropicProvider"
        assert provider.api_key == "test-key"

    def test_create_google_provider(self):
        """Test creating a Google provider."""
        provider = LLMProviderFactory.create_provider("google", api_key="test-key")
        assert provider.__class__.__name__ == "GoogleProvider"
        assert provider.api_key == "test-key"

    def test_create_ollama_provider(self):
        """Test creating an Ollama provider."""
        provider = LLMProviderFactory.create_provider("ollama")
        assert provider.__class__.__name__ == "OllamaProvider"

    def test_case_insensitive_provider_creation(self):
        """Test that provider creation is case insensitive."""
        provider1 = LLMProviderFactory.create_provider("OpenAI", api_key="test")
        provider2 = LLMProviderFactory.create_provider("ANTHROPIC", api_key="test")

        assert provider1.__class__.__name__ == "OpenAIProvider"
        assert provider2.__class__.__name__ == "AnthropicProvider"

    def test_unsupported_provider_raises_error(self):
        """Test that creating an unsupported provider raises an error."""
        with pytest.raises(ValueError, match="Unsupported provider type: unknown"):
            LLMProviderFactory.create_provider("unknown")

    def test_provider_with_additional_kwargs(self):
        """Test creating providers with additional keyword arguments."""
        provider = LLMProviderFactory.create_provider(
            "anthropic", api_key="test-key", model="claude-3-opus-20240229"
        )
        assert provider.__class__.__name__ == "AnthropicProvider"
        assert provider.api_key == "test-key"
        assert provider.default_model == "claude-3-opus-20240229"
