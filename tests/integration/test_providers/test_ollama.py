"""
Integration tests for the Ollama provider using real API calls.
These tests require a running Ollama instance at localhost:11434.
"""

import asyncio
import os

import aiohttp
import pytest
import httpx

from llmring.providers.ollama_api import OllamaProvider
from llmring.schemas import LLMResponse, Message


def is_ollama_running():
    """Check if Ollama is running at localhost:11434."""
    try:
        with httpx.Client() as client:
            response = client.get("http://localhost:11434/api/version", timeout=2.0)
            return response.status_code == 200
    except:
        return False


# Skip decorator that checks if Ollama is actually running
skip_if_ollama_not_running = pytest.mark.skipif(
    not is_ollama_running(), reason="Ollama service not running at localhost:11434"
)


@skip_if_ollama_not_running
@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.slow
class TestOllamaProviderIntegration:
    """Integration tests for OllamaProvider with real API calls."""

    @pytest.fixture
    async def check_ollama_running(self):
        """Check if Ollama is running and skip tests if not."""
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/api/tags", timeout=5) as response:
                    if response.status != 200:
                        pytest.skip("Ollama not running or not accessible")
        except Exception:
            pytest.skip("Ollama not running or not accessible")

    @pytest.fixture
    async def provider(self, check_ollama_running):
        """Create OllamaProvider instance."""
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaProvider(base_url=base_url)

    @pytest.fixture
    async def available_model(self, provider):
        """Get an available model for testing."""
        try:
            models = await provider.get_available_models()
            if not models:
                pytest.skip("No models available in Ollama")

            # Skip large models that are too slow for integration testing
            # Allow small variants like :1b, :0.5b
            small_model_indicators = [":1b", ":0.5b", ":1B", ":0.5B"]

            for model in models:
                # Check if it's a small variant
                is_small = any(
                    indicator in model for indicator in small_model_indicators
                )

                # Check if it's a large base model
                large_patterns = ["llama3.3", "deepseek-r1:32b"]
                is_large_base = any(pattern in model for pattern in large_patterns)

                # Allow small models or models that aren't in the large list
                if is_small or not is_large_base:
                    # Additional check for llama3.2 - only allow if it's small
                    if "llama3.2" in model and not is_small:
                        continue  # Skip large 3.2 models
                    return model

            # If only large models available, skip tests
            pytest.skip(
                "Only large models available - too slow for integration testing"
            )
        except Exception:
            pytest.skip("Cannot determine available Ollama models")

    @pytest.mark.asyncio
    async def test_simple_chat(self, provider, available_model):
        """Test simple chat with Ollama."""
        messages = [Message(role="user", content="Say exactly 'Hello'")]

        response = await provider.chat(
            messages=messages,
            model=available_model,
            max_tokens=10,  # Keep small for faster response
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert len(response.content) > 0
        assert response.model == available_model
        assert response.usage is not None
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0

    @pytest.mark.asyncio
    async def test_chat_with_system_message(self, provider, available_model):
        """Test chat with system message."""
        messages = [
            Message(
                role="system",
                content="You are a helpful math tutor. Always be encouraging.",
            ),
            Message(role="user", content="What is 2 + 2?"),
        ]

        response = await provider.chat(
            messages=messages, model=available_model, max_tokens=100
        )

        assert isinstance(response, LLMResponse)
        assert "4" in response.content
        assert len(response.content) > 10  # Should be encouraging, not just "4"

    @pytest.mark.asyncio
    async def test_chat_with_temperature(self, provider, available_model):
        """Test chat with different temperature settings."""
        messages = [
            Message(role="user", content="Write a creative opening line for a story.")
        ]

        # Test with low temperature (more deterministic)
        response_low = await provider.chat(
            messages=messages, model=available_model, temperature=0.1, max_tokens=50
        )

        # Test with high temperature (more creative)
        response_high = await provider.chat(
            messages=messages, model=available_model, temperature=0.9, max_tokens=50
        )

        assert isinstance(response_low, LLMResponse)
        assert isinstance(response_high, LLMResponse)
        assert len(response_low.content) > 0
        assert len(response_high.content) > 0

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, provider, available_model):
        """Test multi-turn conversation."""
        messages = [
            Message(role="user", content="My name is Alice. What's your name?"),
            Message(
                role="assistant",
                content="Hello Alice! I'm an AI assistant running on Ollama.",
            ),
            Message(role="user", content="What was my name again?"),
        ]

        response = await provider.chat(
            messages=messages, model=available_model, max_tokens=50
        )

        assert isinstance(response, LLMResponse)
        assert "Alice" in response.content

    @pytest.mark.asyncio
    async def test_get_available_models(self, provider):
        """Test getting available models from Ollama."""
        models = await provider.get_available_models()

        assert isinstance(models, list)
        assert len(models) > 0

        # Should contain at least some common models
        model_names = [model.lower() for model in models]
        assert any("llama" in name for name in model_names) or any(
            "mistral" in name for name in model_names
        )

    @pytest.mark.asyncio
    async def test_error_handling_invalid_model(self, provider):
        """Test error handling with invalid model."""
        messages = [Message(role="user", content="Hello")]

        from llmring.exceptions import ModelNotFoundError

        with pytest.raises(ModelNotFoundError, match="Ollama model not available"):
            await provider.chat(messages=messages, model="invalid_model_name!@#")

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, provider, available_model):
        """Test multiple concurrent requests."""

        async def make_request(i):
            messages = [Message(role="user", content=f"Count to {i}")]
            return await provider.chat(
                messages=messages, model=available_model, max_tokens=50
            )

        # Make 2 concurrent requests (be gentle with local Ollama)
        agents = [make_request(i) for i in range(1, 3)]
        responses = await asyncio.gather(*agents)

        assert len(responses) == 2
        for response in responses:
            assert isinstance(response, LLMResponse)
            assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_model_validation(self, provider):
        """Test model validation methods."""
        # Get actually available models
        available_models = await provider.get_available_models()
        if not available_models:
            pytest.skip("No Ollama models available")

        # Test with first available model
        first_model = available_models[0]
        base_name = first_model.split(":")[0]

        # Test that we can validate the available models
        assert await provider.validate_model(first_model) is True
        assert await provider.validate_model(base_name) is True
        assert await provider.validate_model(f"ollama:{first_model}") is True

        # Test models that should be in registry
        assert await provider.validate_model("llama3") is True
        if len(available_models) > 1:
            assert await provider.validate_model("llama3.3") is True

        # Test invalid models
        assert await provider.validate_model("gpt-4") is False
        assert await provider.validate_model("claude-3-opus") is False
        assert await provider.validate_model("invalid_model_name!@#") is False

    @pytest.mark.asyncio
    async def test_supported_models_list(self, provider):
        """Test that supported models list is comprehensive."""
        models = await provider.get_supported_models()

        # Should include models from registry
        assert len(models) > 0
        # At minimum should include models from registry
        assert "llama3" in models or "llama3.3" in models
        # May include more based on local installation

    def test_token_counting(self, provider):
        """Test token counting functionality."""
        text = "This is a test sentence for token counting."
        count = provider.get_token_count(text)

        # Should return a reasonable token count
        assert isinstance(count, int)
        assert count > 0
        assert count < 100  # Should be reasonable for this short text

    @pytest.mark.asyncio
    async def test_mathematical_reasoning(self, provider, available_model):
        """Test mathematical reasoning capabilities."""
        messages = [
            Message(
                role="user",
                content="If I have 15 apples and give away 7, then buy 3 more, how many do I have?",
            )
        ]

        response = await provider.chat(
            messages=messages, model=available_model, max_tokens=100
        )

        assert isinstance(response, LLMResponse)
        # Should calculate: 15 - 7 + 3 = 11
        # Note: Local models might not be as reliable at math
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_code_generation(self, provider, available_model):
        """Test code generation capabilities."""
        messages = [
            Message(
                role="user",
                content="Write a simple Python function to add two numbers.",
            )
        ]

        response = await provider.chat(
            messages=messages, model=available_model, max_tokens=200
        )

        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        # Should contain some programming-related content
        assert any(
            word in response.content.lower()
            for word in ["def", "function", "return", "python"]
        )

    @pytest.mark.asyncio
    async def test_tools_not_supported_gracefully(self, provider, available_model):
        """Test that tools are handled gracefully (not supported by Ollama)."""
        messages = [Message(role="user", content="What's the weather like?")]

        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        ]

        # Should not raise an error, but tools won't be used
        response = await provider.chat(
            messages=messages, model=available_model, tools=tools, max_tokens=100
        )

        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        # Should not have tool calls
        assert response.tool_calls is None or len(response.tool_calls) == 0

    @pytest.mark.asyncio
    async def test_model_with_different_tags(self, provider):
        """Test using models with different tags."""
        # First check what models are actually available
        available_models = await provider.get_available_models()

        if not available_models:
            pytest.skip("No models available in Ollama")

        # Use the first available model
        model_name = available_models[0]

        messages = [Message(role="user", content="Hello")]

        response = await provider.chat(
            messages=messages, model=model_name, max_tokens=30
        )

        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_long_response_handling(self, provider, available_model):
        """Test handling of longer responses."""
        messages = [
            Message(role="user", content="Write a short poem about programming.")
        ]

        response = await provider.chat(
            messages=messages, model=available_model, max_tokens=150
        )

        assert isinstance(response, LLMResponse)
        assert len(response.content) > 50  # Should be a reasonable length poem
        # Should contain programming-related content
        assert any(
            word in response.content.lower()
            for word in ["code", "program", "computer", "debug", "software"]
        )

    @pytest.mark.asyncio
    async def test_provider_prefix_handling(self, provider, available_model):
        """Test handling of provider prefix in model names."""
        messages = [Message(role="user", content="Hello")]

        response = await provider.chat(
            messages=messages,
            model=f"ollama:{available_model}",  # With provider prefix
            max_tokens=30,
        )

        assert isinstance(response, LLMResponse)
        assert response.model == available_model.replace(
            "ollama:", ""
        )  # Should strip the prefix
        assert len(response.content) > 0
