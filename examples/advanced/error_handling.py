"""
Advanced error handling examples for LLMRing.

This example demonstrates:
- Handling specific error types
- Implementing fallback strategies
- Retry logic with exponential backoff
- Error logging and monitoring
"""

import asyncio
import logging

from llmring import LLMRing
from llmring.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from llmring.schemas import LLMRequest, Message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_error_handling():
    """Example 1: Basic error handling."""
    llmring = LLMRing(origin="error-handling-example")

    request = LLMRequest(
        messages=[Message(role="user", content="Hello!")],
        model="openai:gpt-4",
    )

    try:
        response = await llmring.chat(request)
        print(f"Response: {response.content}")

    except ProviderAuthenticationError as e:
        logger.error(f"Authentication failed: {e}")
        logger.info("Check your API key in environment variables")

    except ProviderRateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        logger.info("Wait before retrying or upgrade your plan")

    except ProviderTimeoutError as e:
        logger.error(f"Request timed out: {e}")
        logger.info("Try again or reduce request size")

    except ModelNotFoundError as e:
        logger.error(f"Model not available: {e}")
        logger.info("Check model name or provider status")

    except ProviderResponseError as e:
        logger.error(f"Provider error: {e}")
        logger.info("Check provider status page")

    finally:
        await llmring.close()


async def fallback_providers():
    """Example 2: Fallback to alternative providers."""
    llmring = LLMRing(origin="fallback-example")

    request = LLMRequest(
        messages=[Message(role="user", content="What is 2+2?")],
        model="openai:gpt-4",
    )

    # Try primary provider, fall back to alternatives
    providers = [
        "openai:gpt-4",
        "anthropic:claude-3-sonnet-20240229",
        "google:gemini-pro",
    ]

    for model in providers:
        request.model = model
        try:
            logger.info(f"Trying {model}...")
            response = await llmring.chat(request)
            logger.info(f"Success with {model}")
            print(f"Response: {response.content}")
            break

        except ProviderAuthenticationError:
            logger.warning(f"{model} authentication failed, trying next provider")
            continue

        except ProviderRateLimitError:
            logger.warning(f"{model} rate limited, trying next provider")
            continue

        except Exception as e:
            logger.warning(f"{model} failed: {e}, trying next provider")
            continue

    else:
        logger.error("All providers failed")

    await llmring.close()


async def retry_with_backoff():
    """Example 3: Retry with exponential backoff."""
    llmring = LLMRing(origin="retry-example")

    request = LLMRequest(
        messages=[Message(role="user", content="Hello!")],
        model="openai:gpt-4",
    )

    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = await llmring.chat(request)
            print(f"Response: {response.content}")
            break

        except ProviderRateLimitError:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)  # Exponential backoff
                logger.warning(
                    f"Rate limited, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)
            else:
                logger.error("Max retries exceeded")
                raise

        except ProviderTimeoutError:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"Timeout, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)
            else:
                logger.error("Max retries exceeded")
                raise

    await llmring.close()


async def graceful_degradation():
    """Example 4: Graceful degradation with cheaper models."""
    llmring = LLMRing(origin="degradation-example")

    request = LLMRequest(
        messages=[Message(role="user", content="What is the capital of France?")],
        model="openai:gpt-4",
        max_tokens=100,
    )

    # Try premium model first, fall back to cheaper models
    model_tiers = [
        ("openai:gpt-4", "premium"),
        ("openai:gpt-3.5-turbo", "standard"),
        ("ollama:llama3", "local"),  # Free local model
    ]

    for model, tier in model_tiers:
        request.model = model
        try:
            logger.info(f"Trying {tier} tier ({model})...")
            response = await llmring.chat(request)

            logger.info(f"Success with {tier} tier")
            if response.cost:
                logger.info(f"Cost: ${response.cost.get('total_cost', 0):.4f}")

            print(f"Response: {response.content}")
            break

        except (ProviderRateLimitError, ProviderAuthenticationError):
            logger.warning(f"{tier} tier unavailable, trying next tier")
            continue

        except Exception as e:
            logger.warning(f"{tier} tier failed: {e}")
            continue

    else:
        logger.error("All tiers failed")

    await llmring.close()


async def context_specific_handling():
    """Example 5: Context-specific error handling."""
    llmring = LLMRing(origin="context-example")

    # Simulate a user-facing application
    user_id = "user123"

    request = LLMRequest(
        messages=[Message(role="user", content="Translate 'hello' to French")],
        model="openai:gpt-4",
    )

    try:
        response = await llmring.chat(request)
        return {"success": True, "content": response.content}

    except ProviderAuthenticationError:
        logger.error(f"[{user_id}] Service misconfigured")
        return {
            "success": False,
            "error": "Service temporarily unavailable. Please contact support.",
            "code": "SERVICE_ERROR",
        }

    except ProviderRateLimitError:
        logger.warning(f"[{user_id}] Rate limited")
        return {
            "success": False,
            "error": "Too many requests. Please try again in a moment.",
            "code": "RATE_LIMIT",
        }

    except ProviderTimeoutError:
        logger.warning(f"[{user_id}] Request timeout")
        return {
            "success": False,
            "error": "Request took too long. Please try again.",
            "code": "TIMEOUT",
        }

    except Exception as e:
        logger.error(f"[{user_id}] Unexpected error: {e}")
        return {
            "success": False,
            "error": "An unexpected error occurred. Please try again later.",
            "code": "UNKNOWN_ERROR",
        }

    finally:
        await llmring.close()


async def monitor_errors():
    """Example 6: Error monitoring and metrics."""
    llmring = LLMRing(origin="monitoring-example")

    # Error counters (in production, use proper metrics system)
    error_counts = {
        "auth": 0,
        "rate_limit": 0,
        "timeout": 0,
        "model_not_found": 0,
        "other": 0,
    }

    request = LLMRequest(
        messages=[Message(role="user", content="Hello!")],
        model="openai:gpt-4",
    )

    try:
        response = await llmring.chat(request)
        print(f"Response: {response.content}")

    except ProviderAuthenticationError:
        error_counts["auth"] += 1
        logger.error("Authentication error", extra={"error_type": "auth"})

    except ProviderRateLimitError:
        error_counts["rate_limit"] += 1
        logger.error("Rate limit error", extra={"error_type": "rate_limit"})

    except ProviderTimeoutError:
        error_counts["timeout"] += 1
        logger.error("Timeout error", extra={"error_type": "timeout"})

    except ModelNotFoundError:
        error_counts["model_not_found"] += 1
        logger.error("Model not found", extra={"error_type": "model_not_found"})

    except Exception as e:
        error_counts["other"] += 1
        logger.error(f"Unknown error: {e}", extra={"error_type": "other"})

    finally:
        await llmring.close()

    # Log metrics (in production, send to metrics system)
    logger.info(f"Error counts: {error_counts}")


async def circuit_breaker_pattern():
    """Example 7: Circuit breaker pattern for failing providers."""

    class CircuitBreaker:
        """Simple circuit breaker implementation."""

        def __init__(self, failure_threshold: int = 3, timeout: float = 60.0):
            self.failure_threshold = failure_threshold
            self.timeout = timeout
            self.failures = 0
            self.last_failure_time = 0
            self.state = "closed"  # closed, open, half_open

        def record_success(self):
            """Record successful call."""
            self.failures = 0
            self.state = "closed"

        def record_failure(self):
            """Record failed call."""
            self.failures += 1
            self.last_failure_time = asyncio.get_event_loop().time()

            if self.failures >= self.failure_threshold:
                self.state = "open"
                logger.warning("Circuit breaker opened")

        def is_open(self) -> bool:
            """Check if circuit is open."""
            if self.state == "closed":
                return False

            # Check if timeout passed
            if asyncio.get_event_loop().time() - self.last_failure_time >= self.timeout:
                self.state = "half_open"
                logger.info("Circuit breaker half-open, trying again")
                return False

            return True

    llmring = LLMRing(origin="circuit-breaker-example")
    circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=60.0)

    request = LLMRequest(
        messages=[Message(role="user", content="Hello!")],
        model="openai:gpt-4",
    )

    if circuit_breaker.is_open():
        logger.warning("Circuit breaker is open, skipping request")
        return {"success": False, "error": "Service temporarily unavailable"}

    try:
        response = await llmring.chat(request)
        circuit_breaker.record_success()
        print(f"Response: {response.content}")

    except (ProviderTimeoutError, ProviderResponseError) as e:
        circuit_breaker.record_failure()
        logger.error(f"Request failed: {e}")
        raise

    finally:
        await llmring.close()


# Run examples
if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Basic error handling")
    print("=" * 60)
    asyncio.run(basic_error_handling())

    print("\n" + "=" * 60)
    print("Example 2: Fallback providers")
    print("=" * 60)
    asyncio.run(fallback_providers())

    print("\n" + "=" * 60)
    print("Example 3: Retry with exponential backoff")
    print("=" * 60)
    asyncio.run(retry_with_backoff())

    print("\n" + "=" * 60)
    print("Example 4: Graceful degradation")
    print("=" * 60)
    asyncio.run(graceful_degradation())

    print("\n" + "=" * 60)
    print("Example 7: Circuit breaker pattern")
    print("=" * 60)
    asyncio.run(circuit_breaker_pattern())
