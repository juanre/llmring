"""
Test that all providers properly implement the base interface.

This ensures the base class signature matches provider implementations,
fixing the LSP violation identified in the code review.
"""

import pytest
from unittest.mock import AsyncMock, patch
from typing import AsyncIterator

from llmring.base import BaseLLMProvider
from llmring.schemas import LLMRequest, LLMResponse, Message, StreamChunk
from llmring.providers.openai_api import OpenAIProvider
from llmring.providers.anthropic_api import AnthropicProvider
from llmring.providers.google_api import GoogleProvider
from llmring.providers.ollama_api import OllamaProvider


class TestBaseInterfaceCompliance:
    """Test that providers conform to the base interface."""

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing."""
        return [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello!")
        ]

    @pytest.fixture
    def sample_request(self, sample_messages):
        """Sample LLMRequest for testing."""
        return LLMRequest(
            model="test-model",
            messages=sample_messages,
            temperature=0.7,
            max_tokens=100,
            extra_params={"test_param": "test_value"}
        )

    def test_base_interface_signature(self):
        """Test that base class has the correct signature."""
        import inspect

        sig = inspect.signature(BaseLLMProvider.chat)
        params = list(sig.parameters.keys())

        expected_params = [
            'self', 'messages', 'model', 'temperature', 'max_tokens',
            'response_format', 'tools', 'tool_choice', 'json_response',
            'cache', 'stream', 'extra_params'
        ]

        assert params == expected_params, f"Base interface signature mismatch. Expected {expected_params}, got {params}"

    @pytest.mark.parametrize("provider_class", [
        OpenAIProvider,
        AnthropicProvider,
        GoogleProvider,
        OllamaProvider
    ])
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

    @pytest.mark.asyncio
    async def test_openai_provider_via_base_interface(self, sample_messages):
        """Test calling OpenAI provider through base interface."""
        with patch('llmring.providers.openai_api.AsyncOpenAI') as mock_openai:
            # Mock the API response
            mock_response = AsyncMock()
            mock_response.choices = [AsyncMock()]
            mock_response.choices[0].message.content = "Hello! How can I help you?"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-4"
            mock_response.usage = AsyncMock()
            mock_response.usage.prompt_tokens = 20
            mock_response.usage.completion_tokens = 10
            mock_response.usage.total_tokens = 30

            mock_openai.return_value.chat.completions.create.return_value = mock_response

            # Create provider instance
            provider: BaseLLMProvider = OpenAIProvider(api_key="test-key")

            # Call through base interface
            response = await provider.chat(
                messages=sample_messages,
                model="gpt-4",
                temperature=0.7,
                max_tokens=100,
                extra_params={"logprobs": True}
            )

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello! How can I help you?"
            assert response.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_anthropic_provider_via_base_interface(self, sample_messages):
        """Test calling Anthropic provider through base interface."""
        with patch('llmring.providers.anthropic_api.AsyncAnthropic') as mock_anthropic:
            # Mock the API response
            mock_response = AsyncMock()
            mock_response.content = [AsyncMock()]
            mock_response.content[0].type = "text"
            mock_response.content[0].text = "Hello! How can I assist you?"
            mock_response.model = "claude-3-sonnet-20240229"
            mock_response.stop_reason = "end_turn"
            mock_response.usage = AsyncMock()
            mock_response.usage.input_tokens = 25
            mock_response.usage.output_tokens = 12
            mock_response.usage.cache_creation_input_tokens = 0
            mock_response.usage.cache_read_input_tokens = 0

            mock_anthropic.return_value.messages.create.return_value = mock_response

            # Create provider instance
            provider: BaseLLMProvider = AnthropicProvider(api_key="test-key")

            # Call through base interface
            response = await provider.chat(
                messages=sample_messages,
                model="claude-3-sonnet-20240229",
                temperature=0.5,
                max_tokens=150,
                extra_params={"top_p": 0.9}
            )

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello! How can I assist you?"
            assert response.model == "claude-3-sonnet-20240229"

    @pytest.mark.asyncio
    async def test_streaming_via_base_interface(self, sample_messages):
        """Test streaming through base interface."""
        with patch('llmring.providers.openai_api.AsyncOpenAI') as mock_openai:
            # Mock streaming response
            async def mock_stream():
                yield AsyncMock(
                    choices=[AsyncMock(
                        delta=AsyncMock(content="Hello"),
                        finish_reason=None
                    )]
                )
                yield AsyncMock(
                    choices=[AsyncMock(
                        delta=AsyncMock(content=" there!"),
                        finish_reason="stop"
                    )]
                )

            mock_openai.return_value.chat.completions.create.return_value = mock_stream()

            # Create provider instance
            provider: BaseLLMProvider = OpenAIProvider(api_key="test-key")

            # Call streaming through base interface
            stream = await provider.chat(
                messages=sample_messages,
                model="gpt-4",
                temperature=0.7,
                max_tokens=100,
                stream=True,
                extra_params={"stream": True}
            )

            assert isinstance(stream, AsyncIterator)

            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
                assert isinstance(chunk, StreamChunk)

            assert len(chunks) >= 1
            assert any(chunk.delta for chunk in chunks)

    def test_extra_params_parameter_present(self):
        """Test that all providers accept extra_params parameter."""
        import inspect

        providers = [OpenAIProvider, AnthropicProvider, GoogleProvider, OllamaProvider]

        for provider_class in providers:
            sig = inspect.signature(provider_class.chat)
            assert 'extra_params' in sig.parameters, (
                f"{provider_class.__name__}.chat() missing extra_params parameter"
            )

            # Check it has correct type annotation
            param = sig.parameters['extra_params']
            assert param.default is None, (
                f"{provider_class.__name__} extra_params should default to None"
            )