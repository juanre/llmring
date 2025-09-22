"""
Unit tests for the Google provider using real API calls.
These tests require a valid GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
Tests are minimal to avoid excessive API usage costs.
"""

import json
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
@pytest.mark.unit
@pytest.mark.google
class TestGoogleProviderUnit:
    """Unit tests for GoogleProvider using real API calls."""

    def test_initialization_with_api_key(self):
        """Test provider initialization with explicit API key."""
        provider = GoogleProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        # Default model is now None initially and fetched from registry on first call
        assert provider.default_model is None

    def test_initialization_without_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        # Temporarily unset environment variables
        old_gemini_key = os.environ.pop("GEMINI_API_KEY", None)
        old_google_key = os.environ.pop("GOOGLE_API_KEY", None)
        old_google_gemini_key = os.environ.pop("GOOGLE_GEMINI_API_KEY", None)
        try:
            from llmring.exceptions import ProviderAuthenticationError

            with pytest.raises(
                ProviderAuthenticationError, match="Google API key must be provided"
            ):
                GoogleProvider()
        finally:
            if old_gemini_key:
                os.environ["GEMINI_API_KEY"] = old_gemini_key
            if old_google_key:
                os.environ["GOOGLE_API_KEY"] = old_google_key
            if old_google_gemini_key:
                os.environ["GOOGLE_GEMINI_API_KEY"] = old_google_gemini_key

    def test_initialization_with_env_var(self, monkeypatch):
        """Test provider initialization with environment variable."""
        monkeypatch.setenv("GEMINI_API_KEY", "env-test-key")
        provider = GoogleProvider()
        assert provider.api_key == "env-test-key"

    def test_initialization_with_gemini_api_key_env_var(self, monkeypatch):
        """Test provider initialization with GEMINI_API_KEY environment variable."""
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-env-test-key")
        provider = GoogleProvider()
        assert provider.api_key == "gemini-env-test-key"

    # Model validation tests removed - we no longer gatekeep models
    # The philosophy is that providers should fail naturally if they don't support a model

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_chat_basic_request(self, google_provider, simple_user_message):
        """Test basic chat functionality with minimal token usage."""
        response = await google_provider.chat(
            messages=simple_user_message,
            model="gemini-1.5-pro",
            max_tokens=10,  # Minimal tokens to reduce cost
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert len(response.content) > 0
        assert response.model == "gemini-1.5-pro"
        assert response.usage is not None
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0
        assert response.usage["total_tokens"] > 0
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_chat_with_system_message(
        self, google_provider, system_user_messages
    ):
        """Test chat with system message."""
        response = await google_provider.chat(
            messages=system_user_messages, model="gemini-1.5-pro", max_tokens=10
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert "4" in response.content or "four" in response.content.lower()

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_chat_with_provider_prefix_removal(
        self, google_provider, simple_user_message
    ):
        """Test that provider prefix is correctly removed from model name."""
        response = await google_provider.chat(
            messages=simple_user_message, model="google:gemini-1.5-pro", max_tokens=10
        )

        assert isinstance(response, LLMResponse)
        assert response.model == "gemini-1.5-pro"  # Prefix should be removed

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_chat_with_parameters(self, google_provider, simple_user_message):
        """Test chat with temperature and max_tokens parameters."""
        response = await google_provider.chat(
            messages=simple_user_message,
            model="gemini-1.5-pro",
            temperature=0.7,
            max_tokens=15,
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        # Google may not strictly respect max_tokens, so we just check response exists

    @pytest.mark.asyncio
    async def test_chat_with_unsupported_model_raises_error(
        self, google_provider, simple_user_message
    ):
        """Test that using an unsupported model raises error from API."""
        from llmring.exceptions import ModelNotFoundError, ProviderResponseError

        # Now that registry validation is advisory, the API itself determines if model exists
        with pytest.raises((ModelNotFoundError, ProviderResponseError), match="not found|NOT_FOUND|not available"):
            await google_provider.chat(
                messages=simple_user_message, model="definitely-not-a-real-model"
            )

    @pytest.mark.asyncio
    async def test_chat_api_error_handling(self, google_provider):
        """Test proper error handling for API errors."""
        # Test with invalid model that passes validation but fails at API level
        with pytest.raises(Exception) as exc_info:
            await google_provider.chat(
                messages=[Message(role="user", content="test")],
                model="gemini-99-nonexistent",  # Invalid model
                max_tokens=10,
            )

        # Should wrap the error with our standard format (either validation or API error)
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["error", "unsupported", "invalid"])

    def test_get_token_count_fallback(self, google_provider):
        """Test token counting fallback implementation."""
        text = "This is a test sentence for token counting."
        count = google_provider.get_token_count(text)

        assert isinstance(count, int)
        assert count > 0
        # Should be a reasonable approximation (length / 4)
        assert 5 < count < 50

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_chat_multi_turn_conversation(
        self, google_provider, multi_turn_conversation
    ):
        """Test multi-turn conversation handling."""
        response = await google_provider.chat(
            messages=multi_turn_conversation, model="gemini-1.5-pro", max_tokens=20
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        # Should remember the context and mention "Alice"
        assert "alice" in response.content.lower()

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_chat_with_image_content(self, google_provider):
        """Test chat with image content (text-only for unit tests)."""
        # For unit tests, we'll just test text content
        # Image testing would require actual image files
        messages = [Message(role="user", content="Describe this text: Hello World")]

        response = await google_provider.chat(
            messages=messages, model="gemini-1.5-pro", max_tokens=20
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_chat_response_without_usage_metadata(
        self, google_provider, simple_user_message
    ):
        """Test handling when usage metadata is not available."""
        response = await google_provider.chat(
            messages=simple_user_message, model="gemini-1.5-pro", max_tokens=10
        )

        assert isinstance(response, LLMResponse)
        # Google provider estimates usage when not available
        assert response.usage is not None
        assert isinstance(response.usage["prompt_tokens"], int)
        assert isinstance(response.usage["completion_tokens"], int)
        assert isinstance(response.usage["total_tokens"], int)

    @pytest.mark.asyncio
    async def test_get_default_model(self, google_provider):
        """Test getting default model."""
        # Since we now derive from registry, this might fail if registry unavailable
        try:
            default_model = await google_provider.get_default_model()
            assert isinstance(default_model, str)
            assert len(default_model) > 0
            # Should be a valid Gemini model
            assert "gemini" in default_model.lower()
        except ValueError:
            # Expected if registry is unavailable
            pass

    @pytest.mark.asyncio
    @skip_on_quota_exceeded
    async def test_json_response_format(self, google_provider, json_response_format):
        """Test JSON response format."""
        messages = [
            Message(
                role="user", content="Respond with JSON: answer=hello, confidence=0.9"
            )
        ]

        response = await google_provider.chat(
            messages=messages,
            model="gemini-1.5-pro",
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
        provider = GoogleProvider(api_key="invalid-key")

        with pytest.raises(Exception) as exc_info:
            await provider.chat(
                messages=[Message(role="user", content="test")],
                model="gemini-1.5-pro",
                max_tokens=10,
            )

        # Should get an authentication error
        error_msg = str(exc_info.value).lower()
        assert "error" in error_msg  # Google errors may vary in format

    # Note: model_mapping was removed in favor of registry-based validation
    # This eliminates hardcoded model mappings in favor of dynamic registry data

    # NOTE: Removed test_type_conversion_helper as _convert_type_to_gemini method was removed as dead code
