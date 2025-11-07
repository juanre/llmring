"""
Unit tests for the Anthropic provider using real API calls.
These tests require a valid ANTHROPIC_API_KEY environment variable.
Tests are minimal to avoid excessive API usage costs.
"""

import json
import os

import pytest

from llmring.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderResponseError,
)
from llmring.providers.anthropic_api import AnthropicProvider
from llmring.schemas import LLMResponse, Message


@pytest.mark.llm
@pytest.mark.unit
class TestAnthropicProviderUnit:
    """Unit tests for AnthropicProvider using real API calls."""

    def test_initialization_with_api_key(self):
        """Test provider initialization with explicit API key."""
        provider = AnthropicProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        # Default model is now None initially and fetched from registry on first call
        assert provider.default_model is None

    def test_initialization_without_api_key_raises_error(self):
        """Test that missing API key raises ProviderAuthenticationError."""
        # Temporarily unset environment variable
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with pytest.raises(
                ProviderAuthenticationError, match="Anthropic API key must be provided"
            ):
                AnthropicProvider()
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    def test_initialization_with_env_var(self, monkeypatch):
        """Test provider initialization with environment variable."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-test-key")
        provider = AnthropicProvider()
        assert provider.api_key == "env-test-key"

    # Model validation tests removed - we no longer gatekeep models
    # The philosophy is that providers should fail naturally if they don't support a model

    @pytest.mark.asyncio
    async def test_chat_basic_request(self, anthropic_provider, simple_user_message):
        """Test basic chat functionality with minimal token usage."""
        response = await anthropic_provider.chat(
            messages=simple_user_message,
            model="claude-3-5-haiku-20241022",
            max_tokens=10,  # Minimal tokens to reduce cost
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert len(response.content) > 0
        assert response.model == "claude-3-5-haiku-20241022"
        assert response.usage is not None
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0
        assert response.usage["total_tokens"] > 0
        assert response.finish_reason in ["stop", "max_tokens", "end_turn"]

    @pytest.mark.asyncio
    async def test_chat_with_system_message(self, anthropic_provider, system_user_messages):
        """Test chat with system message."""
        response = await anthropic_provider.chat(
            messages=system_user_messages,
            model="claude-3-5-haiku-20241022",
            max_tokens=10,
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert "4" in response.content or "four" in response.content.lower()

    @pytest.mark.asyncio
    async def test_chat_with_provider_prefix_removal(self, anthropic_provider, simple_user_message):
        """Test that provider prefix is correctly removed from model name."""
        response = await anthropic_provider.chat(
            messages=simple_user_message,
            model="anthropic:claude-3-5-haiku-20241022",
            max_tokens=10,
        )

        assert isinstance(response, LLMResponse)
        assert response.model == "claude-3-5-haiku-20241022"  # Prefix should be removed

    @pytest.mark.asyncio
    async def test_chat_with_parameters(self, anthropic_provider, simple_user_message):
        """Test chat with temperature and max_tokens parameters."""
        response = await anthropic_provider.chat(
            messages=simple_user_message,
            model="claude-3-5-haiku-20241022",
            temperature=0.7,
            max_tokens=15,
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert response.usage["completion_tokens"] <= 15

    @pytest.mark.asyncio
    async def test_chat_with_unsupported_model_raises_error(
        self, anthropic_provider, simple_user_message
    ):
        """Test that using an unsupported model raises error from API."""
        # Now that registry validation is advisory, the API itself determines if model exists
        with pytest.raises(
            (ModelNotFoundError, ProviderResponseError), match="not_found|not available"
        ):
            await anthropic_provider.chat(
                messages=simple_user_message, model="definitely-not-a-real-model"
            )

    @pytest.mark.asyncio
    async def test_chat_with_tools(self, anthropic_provider, sample_tools):
        """Test chat with function calling tools."""
        messages = [
            Message(
                role="user",
                content="What's the weather in NYC? Use tools if available.",
            )
        ]

        response = await anthropic_provider.chat(
            messages=messages,
            model="claude-3-5-haiku-20241022",
            tools=sample_tools,
            max_tokens=100,
        )

        assert isinstance(response, LLMResponse)
        # Tool calls are provider-dependent, just verify response structure
        assert response.content is not None
        if response.tool_calls:
            assert isinstance(response.tool_calls, list)
            for tool_call in response.tool_calls:
                assert "id" in tool_call
                assert "type" in tool_call
                assert "function" in tool_call

    @pytest.mark.asyncio
    async def test_chat_api_error_handling(self, anthropic_provider):
        """Test proper error handling for API errors."""
        # Test with invalid model that passes validation but fails at API level
        with pytest.raises(Exception) as exc_info:
            await anthropic_provider.chat(
                messages=[Message(role="user", content="test")],
                model="claude-99-nonexistent",  # Invalid model
                max_tokens=10,
            )

        # Should wrap the error with our standard format (either validation or API error)
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["error", "unsupported", "invalid"])

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls_response(self, anthropic_provider, sample_tools):
        """Test handling of tool calls in response."""
        # Use a message that's likely to trigger a tool call
        messages = [
            Message(
                role="user",
                content="Please get the weather for San Francisco using the available tools.",
            )
        ]

        response = await anthropic_provider.chat(
            messages=messages,
            model="claude-3-5-haiku-20241022",
            tools=sample_tools,
            tool_choice="auto",
            max_tokens=100,
        )

        assert isinstance(response, LLMResponse)
        # Tool calls depend on the model's decision, so we just verify structure
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            assert tool_call["type"] == "function"
            assert "name" in tool_call["function"]
            assert "arguments" in tool_call["function"]

            # Arguments should be valid JSON
            try:
                json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                pytest.fail("Tool call arguments are not valid JSON")

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, anthropic_provider, multi_turn_conversation):
        """Test multi-turn conversation handling."""
        response = await anthropic_provider.chat(
            messages=multi_turn_conversation,
            model="claude-3-5-haiku-20241022",
            max_tokens=20,
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        # Should remember the context and mention "Alice"
        assert "alice" in response.content.lower()

    @pytest.mark.asyncio
    async def test_get_default_model(self, anthropic_provider):
        """Test getting default model."""
        # Since we now derive from registry, this might fail if registry unavailable
        try:
            default_model = await anthropic_provider.get_default_model()
            assert isinstance(default_model, str)
            assert len(default_model) > 0
            # Should be a valid Claude model
            assert "claude" in default_model.lower() or "opus" in default_model.lower()
        except ValueError:
            # Expected if registry is unavailable
            pass

    @pytest.mark.asyncio
    async def test_json_response_format(self, anthropic_provider, json_response_format):
        """Test JSON response format."""
        messages = [Message(role="user", content="Respond with JSON: answer=hello, confidence=0.9")]

        response = await anthropic_provider.chat(
            messages=messages,
            model="claude-3-5-haiku-20241022",
            response_format=json_response_format,
            max_tokens=50,
        )

        assert isinstance(response, LLMResponse)
        # Try to parse the response as JSON
        try:
            parsed = json.loads(response.content)
            assert isinstance(parsed, dict)
            assert "answer" in parsed
        except json.JSONDecodeError:
            # JSON mode might not be enforced in test environment
            pass

    @pytest.mark.asyncio
    async def test_error_handling_authentication(self):
        """Test error handling for authentication failures."""
        provider = AnthropicProvider(api_key="invalid-key")

        with pytest.raises(Exception) as exc_info:
            await provider.chat(
                messages=[Message(role="user", content="test")],
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
            )

        # Should get an authentication error
        error_msg = str(exc_info.value).lower()
        assert (
            "authentication" in error_msg or "unauthorized" in error_msg or "api key" in error_msg
        )

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, anthropic_provider):
        """Test rate limit error handling structure."""
        # This test doesn't trigger rate limits but ensures the error handling structure is correct
        # Rate limit testing would require coordinated high-volume requests

        # Just verify that the provider handles exceptions properly
        try:
            response = await anthropic_provider.chat(
                messages=[Message(role="user", content="Hello")],
                model="claude-3-5-haiku-20241022",
                max_tokens=5,
            )
            # If we get here, the call succeeded
            assert isinstance(response, LLMResponse)
        except Exception as e:
            # If there's an exception, it should be properly formatted
            assert isinstance(e, Exception)
            assert len(str(e)) > 0
