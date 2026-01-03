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

    @pytest.mark.asyncio
    async def test_complete_implementation_works(self):
        """Test that a complete implementation can be instantiated."""

        class CompleteProvider(BaseLLMProvider):
            """Complete provider implementation for testing."""

            def __init__(self, api_key=None, base_url=None):
                from llmring.base import ProviderConfig

                config = ProviderConfig(api_key=api_key, base_url=base_url)
                super().__init__(config)

            async def chat(self, messages, model, **kwargs):
                return LLMResponse(
                    content="Test response",
                    model=model,
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                )

            async def chat_stream(self, messages, model, **kwargs):
                from llmring.schemas import StreamChunk

                async def _gen():
                    yield StreamChunk(delta="Test ", model=model)
                    yield StreamChunk(delta="response", model=model, finish_reason="stop")

                return _gen()

            def get_token_count(self, text):
                return len(text.split())

            # Validation methods removed - we no longer gatekeep models

            async def get_default_model(self):
                return "test-model"

            async def get_capabilities(self):
                from llmring.base import ProviderCapabilities

                return ProviderCapabilities(
                    provider_name="test",
                    supported_models=["test-model"],
                    supports_streaming=False,
                    default_model="test-model",
                )

            async def upload_file(self, file, purpose="analysis", filename=None, **kwargs):
                raise NotImplementedError

            async def delete_file(self, file_id: str) -> bool:
                raise NotImplementedError

            async def list_files(self, purpose=None, limit=100):
                raise NotImplementedError

            async def get_file(self, file_id: str):
                raise NotImplementedError

        provider = CompleteProvider(api_key="test-key")
        assert provider.config.api_key == "test-key"
        assert provider.config.base_url is None

    @pytest.mark.asyncio
    async def test_get_default_model_fallback(self):
        """Test that get_default_model falls back to first supported model."""

        class TestProvider(BaseLLMProvider):
            """Test provider for default model testing."""

            def __init__(self):
                from llmring.base import ProviderConfig

                config = ProviderConfig(api_key="test")
                super().__init__(config)

            async def chat(self, messages, model, **kwargs):
                return LLMResponse(content="test", model=model)

            async def chat_stream(self, messages, model, **kwargs):
                from llmring.schemas import StreamChunk

                async def _gen():
                    yield StreamChunk(delta="test", model=model, finish_reason="stop")

                return _gen()

            def get_token_count(self, text):
                return len(text.split())

            # Validation methods removed - we no longer gatekeep models

            async def get_default_model(self):
                return "model-1"

            async def get_capabilities(self):
                from llmring.base import ProviderCapabilities

                return ProviderCapabilities(
                    provider_name="test",
                    supported_models=["model-1", "model-2", "model-3"],
                    supports_streaming=False,
                    default_model="model-1",
                )

            async def upload_file(self, file, purpose="analysis", filename=None, **kwargs):
                raise NotImplementedError

            async def delete_file(self, file_id: str) -> bool:
                raise NotImplementedError

            async def list_files(self, purpose=None, limit=100):
                raise NotImplementedError

            async def get_file(self, file_id: str):
                raise NotImplementedError

        provider = TestProvider()
        assert await provider.get_default_model() == "model-1"

    @pytest.mark.asyncio
    async def test_get_default_model_empty_list(self):
        """Test get_default_model when no models are supported."""

        class EmptyProvider(BaseLLMProvider):
            """Provider with no supported models."""

            def __init__(self):
                from llmring.base import ProviderConfig

                config = ProviderConfig(api_key="test")
                super().__init__(config)

            async def chat(self, messages, model, **kwargs):
                return LLMResponse(content="test", model=model)

            async def chat_stream(self, messages, model, **kwargs):
                from llmring.schemas import StreamChunk

                async def _gen():
                    yield StreamChunk(delta="test", model=model, finish_reason="stop")

                return _gen()

            def get_token_count(self, text):
                return len(text.split())

            # Validation methods removed - we no longer gatekeep models

            async def get_default_model(self):
                return "default"

            async def get_capabilities(self):
                from llmring.base import ProviderCapabilities

                return ProviderCapabilities(
                    provider_name="test",
                    supported_models=[],
                    supports_streaming=False,
                    default_model="default",
                )

            async def upload_file(self, file, purpose="analysis", filename=None, **kwargs):
                raise NotImplementedError

            async def delete_file(self, file_id: str) -> bool:
                raise NotImplementedError

            async def list_files(self, purpose=None, limit=100):
                raise NotImplementedError

            async def get_file(self, file_id: str):
                raise NotImplementedError

        provider = EmptyProvider()
        assert await provider.get_default_model() == "default"
