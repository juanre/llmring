"""
Integration tests for the Anthropic provider using real API calls.
These tests require a valid ANTHROPIC_API_KEY environment variable.
"""

import asyncio
import os

import pytest
from llmring.providers.anthropic_api import AnthropicProvider
from llmring.schemas import LLMResponse, Message


@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.slow
class TestAnthropicProviderIntegration:
    """Integration tests for AnthropicProvider with real API calls."""

    @pytest.fixture
    def provider(self):
        """Create AnthropicProvider instance with real API key."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not found in environment")

        return AnthropicProvider(api_key=api_key)

    @pytest.mark.asyncio
    async def test_simple_chat(self, provider):
        """Test simple chat with Claude."""
        messages = [Message(role="user", content="Say exactly 'Hello from Claude!'")]

        response = await provider.chat(
            messages=messages,
            model="claude-3-5-haiku-20241022",  # Use faster/cheaper model for tests
            max_tokens=50,
        )

        assert isinstance(response, LLMResponse)
        assert "Hello from Claude" in response.content
        assert response.model == "claude-3-5-haiku-20241022"
        assert response.usage is not None
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0
        assert response.finish_reason in ["end_turn", "max_tokens"]

    @pytest.mark.asyncio
    async def test_chat_with_system_message(self, provider):
        """Test chat with system message."""
        messages = [
            Message(
                role="system",
                content="You are a helpful math tutor. Always be encouraging.",
            ),
            Message(role="user", content="What is 2 + 2?"),
        ]

        response = await provider.chat(
            messages=messages, model="claude-3-5-haiku-20241022", max_tokens=100
        )

        assert isinstance(response, LLMResponse)
        assert "4" in response.content
        assert len(response.content) > 10  # Should be encouraging, not just "4"

    @pytest.mark.asyncio
    async def test_chat_with_temperature(self, provider):
        """Test chat with different temperature settings."""
        messages = [
            Message(role="user", content="Write a creative opening line for a story.")
        ]

        # Test with low temperature (more deterministic)
        response_low = await provider.chat(
            messages=messages,
            model="claude-3-5-haiku-20241022",
            temperature=0.1,
            max_tokens=50,
        )

        # Test with high temperature (more creative)
        response_high = await provider.chat(
            messages=messages,
            model="claude-3-5-haiku-20241022",
            temperature=0.9,
            max_tokens=50,
        )

        assert isinstance(response_low, LLMResponse)
        assert isinstance(response_high, LLMResponse)
        assert len(response_low.content) > 0
        assert len(response_high.content) > 0

    @pytest.mark.asyncio
    async def test_chat_with_max_tokens(self, provider):
        """Test chat with max_tokens limit."""
        messages = [
            Message(
                role="user", content="Write a long essay about artificial intelligence."
            )
        ]

        response = await provider.chat(
            messages=messages,
            model="claude-3-5-haiku-20241022",
            max_tokens=20,  # Very small limit
        )

        assert isinstance(response, LLMResponse)
        assert response.finish_reason == "max_tokens"
        assert response.usage["completion_tokens"] <= 20

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, provider):
        """Test multi-turn conversation."""
        messages = [
            Message(role="user", content="My name is Alice. What's your name?"),
            Message(
                role="assistant",
                content="Hello Alice! I'm Claude, an AI assistant created by Anthropic.",
            ),
            Message(role="user", content="What was my name again?"),
        ]

        response = await provider.chat(
            messages=messages, model="claude-3-5-haiku-20241022", max_tokens=50
        )

        assert isinstance(response, LLMResponse)
        assert "Alice" in response.content

    @pytest.mark.asyncio
    async def test_chat_with_tools(self, provider):
        """Test chat with function calling."""
        messages = [
            Message(role="user", content="What's the weather like in San Francisco?")
        ]

        tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            }
        ]

        response = await provider.chat(
            messages=messages,
            model="claude-3-5-haiku-20241022",
            tools=tools,
            max_tokens=200,
        )

        assert isinstance(response, LLMResponse)

        # Claude should either use the tool or explain that it can't access real weather data
        if response.tool_calls:
            assert len(response.tool_calls) > 0
            assert response.tool_calls[0]["function"]["name"] == "get_weather"
            # Parse the arguments to ensure they're valid JSON
            import json

            args = json.loads(response.tool_calls[0]["function"]["arguments"])
            assert "location" in args
        else:
            # If no tool call, should mention inability to access real weather
            assert any(
                word in response.content.lower()
                for word in ["cannot", "can't", "unable", "don't have access"]
            )

    @pytest.mark.asyncio
    async def test_error_handling_invalid_model(self, provider):
        """Test error handling with invalid model."""
        messages = [Message(role="user", content="Hello")]

        with pytest.raises(ValueError, match="Unsupported model"):
            await provider.chat(messages=messages, model="invalid-model-name")

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, provider):
        """Test multiple concurrent requests."""

        async def make_request(i):
            messages = [Message(role="user", content=f"Count to {i}")]
            return await provider.chat(
                messages=messages, model="claude-3-5-haiku-20241022", max_tokens=50
            )

        # Make 3 concurrent requests
        agents = [make_request(i) for i in range(1, 4)]
        responses = await asyncio.gather(*agents)

        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, LLMResponse)
            assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_model_validation(self, provider):
        """Test model validation methods."""
        # Test valid models
        assert provider.validate_model("claude-3-5-haiku-20241022") is True
        assert provider.validate_model("claude-3-5-sonnet-20241022") is True
        assert provider.validate_model("anthropic:claude-3-5-haiku-20241022") is True

        # Test invalid models
        assert provider.validate_model("gpt-4") is False
        assert provider.validate_model("invalid-model") is False

    def test_supported_models_list(self, provider):
        """Test that supported models list is comprehensive."""
        models = provider.get_supported_models()

        # Should include Claude 3.7 models
        assert "claude-3-7-sonnet-20250219" in models
        assert "claude-3-7-sonnet" in models

        # Should include Claude 3.5 models
        assert "claude-3-5-sonnet-20241022" in models
        assert "claude-3-5-haiku-20241022" in models

        # Should include Claude 3 models
        assert "claude-3-opus-20240229" in models
        assert "claude-3-sonnet-20240229" in models
        assert "claude-3-haiku-20240307" in models

    def test_token_counting(self, provider):
        """Test token counting functionality."""
        text = "This is a test sentence for token counting."
        count = provider.get_token_count(text)

        # Should return a reasonable token count
        assert isinstance(count, int)
        assert count > 0
        assert count < 100  # Should be reasonable for this short text

    @pytest.mark.asyncio
    async def test_json_mode_instruction(self, provider):
        """Test JSON mode handling (through instructions since Claude doesn't have native JSON mode)."""
        messages = [Message(role="user", content="List three colors as a JSON array.")]

        response = await provider.chat(
            messages=messages,
            model="claude-3-5-haiku-20241022",
            response_format={"type": "json_object"},
            max_tokens=100,
        )

        assert isinstance(response, LLMResponse)

        # Response should contain JSON
        try:
            import json

            # Try to parse response as JSON (might be wrapped in explanation)
            if "[" in response.content:
                json_start = response.content.find("[")
                json_end = response.content.rfind("]") + 1
                json_part = response.content[json_start:json_end]
                parsed = json.loads(json_part)
                assert isinstance(parsed, list)
        except json.JSONDecodeError:
            # If not valid JSON, should at least mention JSON in the response
            assert "json" in response.content.lower() or "[" in response.content
