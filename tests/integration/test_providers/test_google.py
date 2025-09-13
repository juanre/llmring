"""
Integration tests for the Google provider using real API calls.
These tests require a valid GOOGLE_API_KEY or GEMINI_API_KEY environment variable.
"""

import asyncio
import os
from functools import wraps

import pytest

from llmring.providers.google_api import GoogleProvider
from llmring.schemas import LLMResponse, Message


def skip_on_quota_exceeded(func):
    """Decorator to skip tests when Google API quota is exceeded."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            # Check for quota/rate limit errors
            if any(
                term in error_msg
                for term in [
                    "quota",
                    "rate limit",
                    "resource_exhausted",
                    "429",
                    "billing",
                    "exceeded",
                ]
            ):
                pytest.skip(f"Google API quota exceeded: {str(e)[:100]}")
            raise

    return wrapper


@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.google
class TestGoogleProviderIntegration:
    """Integration tests for GoogleProvider with real API calls."""

    @pytest.fixture
    def provider(self):
        """Create GoogleProvider instance with real API key."""
        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GOOGLE_GEMINI_API_KEY")
        )
        if not api_key:
            pytest.skip("GOOGLE_API_KEY, GEMINI_API_KEY, or GOOGLE_GEMINI_API_KEY not found in environment")

        return GoogleProvider(api_key=api_key)

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_simple_chat(self, provider):
        """Test simple chat with Gemini."""
        messages = [Message(role="user", content="Say exactly 'Hello from Gemini!'")]

        response = await provider.chat(
            messages=messages,
            model="gemini-1.5-flash",  # Use faster model for tests
            max_tokens=50,
        )

        assert isinstance(response, LLMResponse)
        assert "Hello from Gemini" in response.content
        assert response.model == "gemini-1.5-flash"

        # Google API might not always provide usage metadata
        if response.usage:
            assert response.usage["prompt_tokens"] > 0
            assert response.usage["completion_tokens"] > 0

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
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
            messages=messages, model="gemini-1.5-flash", max_tokens=100
        )

        assert isinstance(response, LLMResponse)
        assert "4" in response.content
        assert len(response.content) > 10  # Should be encouraging, not just "4"

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_chat_with_temperature(self, provider):
        """Test chat with different temperature settings."""
        messages = [
            Message(role="user", content="Write a creative opening line for a story.")
        ]

        # Test with low temperature (more deterministic)
        response_low = await provider.chat(
            messages=messages, model="gemini-1.5-flash", temperature=0.1, max_tokens=50
        )

        # Test with high temperature (more creative)
        response_high = await provider.chat(
            messages=messages, model="gemini-1.5-flash", temperature=0.9, max_tokens=50
        )

        assert isinstance(response_low, LLMResponse)
        assert isinstance(response_high, LLMResponse)
        assert len(response_low.content) > 0
        assert len(response_high.content) > 0

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_chat_with_max_tokens(self, provider):
        """Test chat with max_tokens limit."""
        messages = [
            Message(
                role="user", content="Write a long essay about artificial intelligence."
            )
        ]

        response = await provider.chat(
            messages=messages,
            model="gemini-1.5-flash",
            max_tokens=20,  # Very small limit
        )

        assert isinstance(response, LLMResponse)
        # Google might handle token limits differently than other providers
        assert len(response.content) > 0

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_multi_turn_conversation(self, provider):
        """Test multi-turn conversation."""
        messages = [
            Message(role="user", content="My name is Alice. What's your name?"),
            Message(
                role="assistant",
                content="Hello Alice! I'm Gemini, a large language model from Google.",
            ),
            Message(role="user", content="What was my name again?"),
        ]

        response = await provider.chat(
            messages=messages, model="gemini-1.5-flash", max_tokens=50
        )

        assert isinstance(response, LLMResponse)
        assert "Alice" in response.content

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_error_handling_invalid_model(self, provider):
        """Test error handling with invalid model."""
        messages = [Message(role="user", content="Hello")]

        from llmring.exceptions import ModelNotFoundError
        with pytest.raises(ModelNotFoundError, match="Unsupported model"):
            await provider.chat(messages=messages, model="invalid-model-name")

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_concurrent_requests(self, provider):
        """Test multiple concurrent requests."""

        async def make_request(i):
            messages = [Message(role="user", content=f"Count to {i}")]
            return await provider.chat(
                messages=messages, model="gemini-1.5-flash", max_tokens=50
            )

        # Make 3 concurrent requests
        agents = [make_request(i) for i in range(1, 4)]
        responses = await asyncio.gather(*agents)

        assert len(responses) == 3
        for response in responses:
            assert isinstance(response, LLMResponse)
            assert len(response.content) > 0

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_model_validation(self, provider):
        """Test model validation methods."""
        # Test valid models
        assert provider.validate_model("gemini-1.5-pro") is True
        assert provider.validate_model("gemini-1.5-flash") is True
        assert provider.validate_model("google:gemini-1.5-pro") is True

        # Test invalid models
        assert provider.validate_model("gpt-4") is False
        assert provider.validate_model("claude-3-opus") is False
        assert provider.validate_model("invalid-model") is False

    def test_supported_models_list(self, provider):
        """Test that supported models list is comprehensive."""
        models = provider.get_supported_models()

        # Should include Gemini models
        assert "gemini-1.5-pro" in models
        assert "gemini-1.5-flash" in models
        assert "gemini-pro" in models

    def test_token_counting(self, provider):
        """Test token counting functionality."""
        text = "This is a test sentence for token counting."
        count = provider.get_token_count(text)

        # Should return a reasonable token count
        assert isinstance(count, int)
        assert count > 0
        assert count < 100  # Should be reasonable for this short text

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_chat_with_image_content(self, provider):
        """Test chat with image content (vision)."""
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "What color is this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                        },
                    },
                ],
            )
        ]

        try:
            response = await provider.chat(
                messages=messages,
                model="gemini-1.5-pro",  # Use Pro for vision agents
                max_tokens=100,
            )

            assert isinstance(response, LLMResponse)
            assert len(response.content) > 0
            # Should mention something about color (it's a blue pixel)

        except Exception as e:
            # Vision might not be available or might have restrictions
            if "quota" in str(e).lower() or "billing" in str(e).lower():
                pytest.skip(f"Vision API not available: {e}")
            else:
                raise

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_long_context_handling(self, provider):
        """Test handling of long context (Gemini has very large context window)."""
        # Create a longer conversation
        long_text = "This is a test sentence. " * 100  # Repeat 100 times
        messages = [
            Message(role="user", content=f"Here's some text: {long_text}"),
            Message(role="user", content="What was the first word in that text?"),
        ]

        try:
            response = await provider.chat(
                messages=messages, model="gemini-1.5-flash", max_tokens=50
            )

            assert isinstance(response, LLMResponse)
            # Should be able to identify the first word despite the long context
            assert "This" in response.content or "first" in response.content.lower()
        except Exception as e:
            # Handle rate limiting or quota issues
            if any(
                keyword in str(e).lower()
                for keyword in ["quota", "rate", "billing", "resource_exhausted", "429"]
            ):
                pytest.skip(f"Google API rate limit or quota exceeded: {e}")
            else:
                raise

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_mathematical_reasoning(self, provider):
        """Test mathematical reasoning capabilities."""
        messages = [
            Message(
                role="user",
                content="If I have 15 apples and give away 7, then buy 3 more, how many do I have?",
            )
        ]

        try:
            response = await provider.chat(
                messages=messages, model="gemini-1.5-flash", max_tokens=100
            )

            assert isinstance(response, LLMResponse)
            # Should calculate: 15 - 7 + 3 = 11
            assert "11" in response.content
        except Exception as e:
            # Handle rate limiting or quota issues
            if any(
                keyword in str(e).lower()
                for keyword in ["quota", "rate", "billing", "resource_exhausted", "429"]
            ):
                pytest.skip(f"Google API rate limit or quota exceeded: {e}")
            else:
                raise

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_code_generation(self, provider):
        """Test code generation capabilities."""
        messages = [
            Message(
                role="user",
                content="Write a simple Python function to add two numbers.",
            )
        ]

        try:
            response = await provider.chat(
                messages=messages, model="gemini-1.5-flash", max_tokens=200
            )

            assert isinstance(response, LLMResponse)
            assert "def" in response.content
            assert "return" in response.content
            assert "+" in response.content or "add" in response.content.lower()
        except Exception as e:
            # Handle rate limiting or quota issues
            if any(
                keyword in str(e).lower()
                for keyword in ["quota", "rate", "billing", "resource_exhausted", "429"]
            ):
                pytest.skip(f"Google API rate limit or quota exceeded: {e}")
            else:
                raise

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_safety_filtering(self, provider):
        """Test that safety filtering works appropriately."""
        messages = [
            Message(role="user", content="Tell me a wholesome joke about programming.")
        ]

        try:
            response = await provider.chat(
                messages=messages, model="gemini-1.5-flash", max_tokens=200
            )

            assert isinstance(response, LLMResponse)
            assert len(response.content) > 0
            # Should contain programming-related content
            assert any(
                word in response.content.lower()
                for word in ["code", "program", "bug", "debug", "computer"]
            )
        except Exception as e:
            # Handle rate limiting or quota issues
            if any(
                keyword in str(e).lower()
                for keyword in ["quota", "rate", "billing", "resource_exhausted", "429"]
            ):
                pytest.skip(f"Google API rate limit or quota exceeded: {e}")
            else:
                raise
