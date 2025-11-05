"""Unit tests for ProviderErrorHandler."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmring.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from llmring.net.retry import RetryError
from llmring.providers.error_handler import ProviderErrorHandler


class TestProviderErrorHandler:
    """Tests for ProviderErrorHandler."""

    @pytest.fixture
    def breaker(self):
        """Create a mock circuit breaker."""
        breaker = MagicMock()
        breaker.record_failure = AsyncMock()
        return breaker

    @pytest.fixture
    def handler(self, breaker):
        """Create an error handler instance."""
        return ProviderErrorHandler(provider_name="test_provider", breaker=breaker)

    @pytest.mark.asyncio
    async def test_already_wrapped_error_reraises(self, handler):
        """Should re-raise if exception is already LLMRingError."""
        wrapped_error = ModelNotFoundError(
            "Already wrapped", provider="test", model_name="test-model"
        )

        with pytest.raises(ModelNotFoundError) as exc_info:
            await handler.handle_error(wrapped_error, "test-model")

        assert exc_info.value is wrapped_error

    @pytest.mark.asyncio
    async def test_records_failure_with_breaker(self, handler, breaker):
        """Should record failure in circuit breaker."""
        try:
            await handler.handle_error(Exception("test"), "test-model")
        except Exception:
            pass

        breaker.record_failure.assert_called_once_with("test_provider:test-model")

    @pytest.mark.asyncio
    async def test_breaker_failure_does_not_propagate(self, handler, breaker):
        """Should not propagate circuit breaker failures."""
        breaker.record_failure.side_effect = Exception("breaker failed")

        with pytest.raises(ProviderResponseError):
            await handler.handle_error(Exception("test"), "test-model")

        # Should complete without raising the breaker exception

    @pytest.mark.asyncio
    async def test_no_breaker_still_works(self):
        """Should work without a circuit breaker."""
        handler = ProviderErrorHandler(provider_name="test", breaker=None)

        with pytest.raises(ProviderResponseError):
            await handler.handle_error(Exception("test"), "test-model")

    @pytest.mark.asyncio
    async def test_direct_timeout_error(self, handler):
        """Should handle direct asyncio.TimeoutError."""
        with pytest.raises(ProviderTimeoutError) as exc_info:
            await handler.handle_error(asyncio.TimeoutError(), "test-model")

        assert exc_info.value.provider == "test_provider"
        assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retry_timeout_error(self, handler):
        """Should handle timeout after retries."""
        retry_error = RetryError("Retry failed", attempts=3)
        retry_error.__cause__ = asyncio.TimeoutError()

        with pytest.raises(ProviderTimeoutError) as exc_info:
            await handler.handle_error(retry_error, "test-model")

        assert "after 3 attempts" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_connection_error(self, handler):
        """Should handle connection errors."""
        with pytest.raises(ProviderResponseError) as exc_info:
            await handler.handle_error(ConnectionError(), "test-model")

        assert "Cannot connect" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_os_error_treated_as_connection_error(self, handler):
        """Should handle OSError as connection error."""
        with pytest.raises(ProviderResponseError) as exc_info:
            await handler.handle_error(OSError(), "test-model")

        assert "Cannot connect" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_context_included_in_messages(self, handler):
        """Should include context in error messages when provided."""
        with pytest.raises(ProviderResponseError) as exc_info:
            await handler.handle_error(Exception("test"), "test-model", context="streaming")

        assert "(streaming)" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_openai_not_found_error(self):
        """Should detect OpenAI NotFoundError."""
        handler = ProviderErrorHandler("openai")

        # Create a mock exception that looks like OpenAI's NotFoundError
        class NotFoundError(Exception):
            pass

        with pytest.raises(ModelNotFoundError) as exc_info:
            await handler.handle_error(NotFoundError(), "gpt-4")

        assert exc_info.value.model_name == "gpt-4"
        assert exc_info.value.provider == "openai"

    @pytest.mark.asyncio
    async def test_openai_authentication_error(self):
        """Should detect OpenAI AuthenticationError."""
        handler = ProviderErrorHandler("openai")

        class AuthenticationError(Exception):
            pass

        with pytest.raises(ProviderAuthenticationError):
            await handler.handle_error(AuthenticationError(), "gpt-4")

    @pytest.mark.asyncio
    async def test_openai_rate_limit_error(self):
        """Should detect OpenAI RateLimitError."""
        handler = ProviderErrorHandler("openai")

        class RateLimitError(Exception):
            retry_after = 60

        error = RateLimitError()

        with pytest.raises(ProviderRateLimitError) as exc_info:
            await handler.handle_error(error, "gpt-4")

        assert exc_info.value.retry_after == 60

    @pytest.mark.asyncio
    async def test_anthropic_not_found_error(self):
        """Should detect Anthropic NotFoundError."""
        handler = ProviderErrorHandler("anthropic")

        class NotFoundError(Exception):
            pass

        with pytest.raises(ModelNotFoundError):
            await handler.handle_error(NotFoundError(), "claude-3")

    @pytest.mark.asyncio
    async def test_anthropic_bad_request_with_model(self):
        """Should detect Anthropic BadRequestError mentioning model."""
        handler = ProviderErrorHandler("anthropic")

        class BadRequestError(Exception):
            pass

        with pytest.raises(ModelNotFoundError):
            await handler.handle_error(BadRequestError("model not supported"), "claude-3")

    @pytest.mark.asyncio
    async def test_anthropic_bad_request_without_model(self):
        """Should handle Anthropic BadRequestError not about model."""
        handler = ProviderErrorHandler("anthropic")

        class BadRequestError(Exception):
            pass

        with pytest.raises(ProviderResponseError):
            await handler.handle_error(BadRequestError("invalid parameter"), "claude-3")

    @pytest.mark.asyncio
    async def test_google_model_not_found_error(self):
        """Should detect Google model not found from error message."""
        handler = ProviderErrorHandler("google")

        with pytest.raises(ModelNotFoundError):
            await handler.handle_error(Exception("model not found"), "gemini-pro")

    @pytest.mark.asyncio
    async def test_google_model_not_supported_error(self):
        """Should detect Google model not supported from error message."""
        handler = ProviderErrorHandler("google")

        with pytest.raises(ModelNotFoundError):
            await handler.handle_error(Exception("model not supported"), "gemini-pro")

    @pytest.mark.asyncio
    async def test_google_rate_limit_error(self):
        """Should detect Google rate limit from error message."""
        handler = ProviderErrorHandler("google")

        with pytest.raises(ProviderRateLimitError):
            await handler.handle_error(Exception("rate limit exceeded"), "gemini-pro")

    @pytest.mark.asyncio
    async def test_google_quota_error(self):
        """Should detect Google quota error as rate limit."""
        handler = ProviderErrorHandler("google")

        with pytest.raises(ProviderRateLimitError):
            await handler.handle_error(Exception("quota exceeded"), "gemini-pro")

    @pytest.mark.asyncio
    async def test_google_authentication_error(self):
        """Should detect Google authentication from error message."""
        handler = ProviderErrorHandler("google")

        with pytest.raises(ProviderAuthenticationError):
            await handler.handle_error(Exception("api key invalid"), "gemini-pro")

    @pytest.mark.asyncio
    async def test_google_unauthorized_error(self):
        """Should detect Google unauthorized as authentication error."""
        handler = ProviderErrorHandler("google")

        with pytest.raises(ProviderAuthenticationError):
            await handler.handle_error(Exception("unauthorized"), "gemini-pro")

    @pytest.mark.asyncio
    async def test_ollama_response_error(self):
        """Should detect Ollama ResponseError."""
        handler = ProviderErrorHandler("ollama")

        class ResponseError(Exception):
            def __init__(self, msg):
                self.error = msg

        with pytest.raises(ModelNotFoundError):
            await handler.handle_error(ResponseError("model 'llama2' not found"), "llama2")

    @pytest.mark.asyncio
    async def test_retry_error_with_openai_exception(self):
        """Should handle RetryError wrapping OpenAI exception."""
        handler = ProviderErrorHandler("openai")

        class NotFoundError(Exception):
            pass

        retry_error = RetryError("Failed", attempts=3)
        retry_error.__cause__ = NotFoundError()

        with pytest.raises(ModelNotFoundError) as exc_info:
            await handler.handle_error(retry_error, "gpt-4")

        assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retry_error_with_anthropic_rate_limit(self):
        """Should handle RetryError wrapping Anthropic rate limit."""
        handler = ProviderErrorHandler("anthropic")

        class RateLimitError(Exception):
            retry_after = 120

        retry_error = RetryError("Failed", attempts=5)
        retry_error.__cause__ = RateLimitError()

        with pytest.raises(ProviderRateLimitError) as exc_info:
            await handler.handle_error(retry_error, "claude-3")

        assert exc_info.value.retry_after == 120

    @pytest.mark.asyncio
    async def test_retry_error_generic_failure(self):
        """Should handle generic retry failure."""
        handler = ProviderErrorHandler("openai")

        retry_error = RetryError("Failed", attempts=3)
        retry_error.__cause__ = Exception("unknown error")

        with pytest.raises(ProviderResponseError) as exc_info:
            await handler.handle_error(retry_error, "gpt-4")

        assert "after 3 attempts" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unknown_exception_wrapped(self, handler):
        """Should wrap unknown exceptions."""
        with pytest.raises(ProviderResponseError) as exc_info:
            await handler.handle_error(ValueError("unknown"), "test-model")

        assert "Unexpected error" in str(exc_info.value)
        assert isinstance(exc_info.value.original, ValueError)

    @pytest.mark.asyncio
    async def test_extract_error_message_with_message_attr(self, handler):
        """Should extract error from message attribute."""

        class CustomError(Exception):
            message = "custom error message"

        error_type = handler._detect_error_type(CustomError())
        # Should not crash and should fall back to generic handling
        assert error_type is None

    @pytest.mark.asyncio
    async def test_extract_error_message_with_error_attr(self, handler):
        """Should extract error from error attribute."""

        class CustomError(Exception):
            error = "custom error detail"

        error_type = handler._detect_error_type(CustomError())
        assert error_type is None

    @pytest.mark.asyncio
    async def test_extract_error_message_fallback_to_str(self, handler):
        """Should fall back to str() for error message."""
        error_msg = handler._extract_error_message(Exception("fallback message"))
        assert error_msg == "fallback message"
