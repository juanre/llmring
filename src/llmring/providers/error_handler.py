"""
Unified error handling for all LLM providers.

Provides consistent exception mapping from provider-specific errors
to llmring's unified exception hierarchy.
"""

import asyncio
import logging
from typing import Any, Optional, Protocol

from llmring.exceptions import (
    LLMRingError,
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)


class CircuitBreakerProtocol(Protocol):
    """Protocol for circuit breaker dependencies."""

    async def record_failure(self, key: str) -> None:
        """Record a failure for the given key."""
        ...


class ProviderErrorHandler:
    """
    Unified error handling for all providers.

    Handles:
    - Retry error unwrapping
    - Timeout detection
    - Rate limiting
    - Authentication failures
    - Model not found errors
    - Generic error wrapping

    Each provider can customize behavior via strategy pattern.
    """

    def __init__(
        self,
        provider_name: str,
        breaker: Optional[CircuitBreakerProtocol] = None,
    ):
        """
        Initialize the error handler.

        Args:
            provider_name: Name of the provider (e.g., "openai", "anthropic")
            breaker: Optional circuit breaker for recording failures
        """
        self.provider_name = provider_name
        self.breaker = breaker

    async def handle_error(
        self,
        exception: Exception,
        model: str,
        *,
        context: Optional[str] = None,
    ) -> None:
        """
        Handle an exception from a provider API call.

        This method never returns - it always raises a mapped exception.

        Args:
            exception: The caught exception
            model: The model name being used
            context: Optional context string for error messages

        Raises:
            LLMRingError: Always raises a mapped exception
        """
        # Already wrapped? Just re-raise
        if isinstance(exception, LLMRingError):
            raise exception

        # Record failure (non-blocking)
        if self.breaker:
            try:
                await self.breaker.record_failure(f"{self.provider_name}:{model}")
            except Exception:
                pass

        # Check for RetryError wrapper first
        from llmring.net.retry import RetryError

        if isinstance(exception, RetryError):
            await self._handle_retry_error(exception, model, context)

        # Handle direct exceptions
        await self._handle_direct_error(exception, model, context)

    async def _handle_retry_error(
        self,
        exception: Exception,
        model: str,
        context: Optional[str],
    ) -> None:
        """Handle errors wrapped in RetryError."""
        root = exception.__cause__ if hasattr(exception, "__cause__") else exception
        attempts = getattr(exception, "attempts", 0)
        context_str = f" ({context})" if context else ""

        # Timeout after retries
        if isinstance(root, asyncio.TimeoutError):
            raise ProviderTimeoutError(
                f"Request timed out after {attempts} attempts{context_str}",
                provider=self.provider_name,
                original=exception,
            ) from exception

        # Provider-specific error detection
        error_type = self._detect_error_type(root)

        if error_type == "model_not_found":
            raise ModelNotFoundError(
                f"Model '{model}' not found{context_str}",
                provider=self.provider_name,
                model_name=model,
                original=exception,
            ) from exception
        elif error_type == "authentication":
            raise ProviderAuthenticationError(
                f"Authentication failed{context_str}",
                provider=self.provider_name,
                original=exception,
            ) from exception
        elif error_type == "rate_limit":
            raise ProviderRateLimitError(
                f"Rate limit exceeded{context_str}",
                provider=self.provider_name,
                retry_after=getattr(root, "retry_after", None),
                original=exception,
            ) from exception

        # Generic retry exhaustion
        raise ProviderResponseError(
            f"Request failed after {attempts} attempts{context_str}",
            provider=self.provider_name,
            original=exception,
        ) from exception

    async def _handle_direct_error(
        self,
        exception: Exception,
        model: str,
        context: Optional[str],
    ) -> None:
        """Handle direct (non-retry) exceptions."""
        context_str = f" ({context})" if context else ""

        # Direct timeout
        if isinstance(exception, asyncio.TimeoutError):
            raise ProviderTimeoutError(
                f"Request timed out{context_str}",
                provider=self.provider_name,
                original=exception,
            ) from exception

        # Connection errors (mainly for Ollama)
        if isinstance(exception, (ConnectionError, OSError)):
            raise ProviderResponseError(
                f"Cannot connect to {self.provider_name}{context_str}",
                provider=self.provider_name,
                original=exception,
            ) from exception

        # Provider-specific error detection
        error_type = self._detect_error_type(exception)

        if error_type == "model_not_found":
            raise ModelNotFoundError(
                f"Model '{model}' not available{context_str}",
                provider=self.provider_name,
                model_name=model,
                original=exception,
            ) from exception
        elif error_type == "authentication":
            raise ProviderAuthenticationError(
                f"Authentication failed{context_str}",
                provider=self.provider_name,
                original=exception,
            ) from exception
        elif error_type == "rate_limit":
            raise ProviderRateLimitError(
                f"Rate limit exceeded{context_str}",
                provider=self.provider_name,
                retry_after=getattr(exception, "retry_after", None),
                original=exception,
            ) from exception
        elif error_type == "bad_request":
            raise ProviderResponseError(
                f"Bad request{context_str}",
                provider=self.provider_name,
                original=exception,
            ) from exception

        # Unknown error - wrap minimally
        raise ProviderResponseError(
            f"Unexpected error{context_str}",
            provider=self.provider_name,
            original=exception,
        ) from exception

    def _detect_error_type(self, exception: Exception) -> Optional[str]:
        """
        Detect the type of error from an exception.

        Returns one of:
        - "model_not_found"
        - "authentication"
        - "rate_limit"
        - "bad_request"
        - None (unknown)
        """
        exception_type = type(exception).__name__

        # Provider-specific exception detection
        if self.provider_name in ("openai", "anthropic"):
            return self._detect_openai_anthropic_error(exception, exception_type)
        elif self.provider_name == "google":
            return self._detect_google_error(exception, exception_type)
        elif self.provider_name == "ollama":
            return self._detect_ollama_error(exception, exception_type)

        return None

    def _detect_openai_anthropic_error(
        self, exception: Exception, exception_type: str
    ) -> Optional[str]:
        """Detect errors for OpenAI and Anthropic SDKs."""
        # These providers have explicit exception classes
        if "NotFoundError" in exception_type:
            return "model_not_found"
        elif "AuthenticationError" in exception_type:
            return "authentication"
        elif "RateLimitError" in exception_type:
            return "rate_limit"
        elif "BadRequestError" in exception_type:
            # Special handling for Anthropic's BadRequestError with model issues
            if self.provider_name == "anthropic" and "model" in str(exception).lower():
                return "model_not_found"
            return "bad_request"

        return None

    def _detect_google_error(self, exception: Exception, exception_type: str) -> Optional[str]:
        """Detect errors for Google SDK."""
        # Google uses string matching
        error_msg = self._extract_error_message(exception)

        if "model" in error_msg.lower() and (
            "not found" in error_msg.lower() or "not supported" in error_msg.lower()
        ):
            return "model_not_found"
        elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
            return "rate_limit"
        elif "api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return "authentication"

        return None

    def _detect_ollama_error(self, exception: Exception, exception_type: str) -> Optional[str]:
        """Detect errors for Ollama."""
        # Ollama uses ResponseError with error messages
        if "ResponseError" in exception_type:
            error_msg = str(getattr(exception, "error", exception))
            if "model" in error_msg.lower():
                return "model_not_found"

        return None

    def _extract_error_message(self, exception: Exception) -> str:
        """
        Extract a readable error message from an exception.

        Tries multiple attributes commonly used by SDKs.
        """
        # Try common attributes
        for attr in ("message", "error", "detail", "msg"):
            if hasattr(exception, attr):
                msg = getattr(exception, attr)
                if msg:
                    return str(msg)

        # Fall back to string representation
        return str(exception)
