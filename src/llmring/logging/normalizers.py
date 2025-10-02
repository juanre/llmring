"""
Provider detection and response normalization for decorator-based logging.

Supports auto-detection and normalization of responses from:
- OpenAI SDK
- Anthropic SDK
- Google Gemini SDK
"""

import inspect
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def detect_provider(response: Any, func: Any = None) -> Optional[str]:
    """
    Auto-detect provider from response object or function.

    Args:
        response: The response object from the LLM SDK
        func: Optional function that was called

    Returns:
        Provider name ("openai", "anthropic", "google") or None if unknown
    """
    # Detect from response object type
    response_type = type(response).__name__
    response_module = type(response).__module__

    # OpenAI detection
    if "openai" in response_module.lower():
        return "openai"
    if response_type in ("ChatCompletion", "Completion"):
        return "openai"

    # Anthropic detection
    if "anthropic" in response_module.lower():
        return "anthropic"
    if response_type == "Message" and hasattr(response, "content") and hasattr(response, "stop_reason"):
        return "anthropic"

    # Google Gemini detection
    if "google" in response_module.lower() or "genai" in response_module.lower():
        return "google"
    if response_type in ("GenerateContentResponse", "ChatResponse"):
        return "google"

    # Detect from function module name if available
    if func:
        func_module = inspect.getmodule(func)
        if func_module:
            module_name = func_module.__name__.lower()
            if "openai" in module_name:
                return "openai"
            if "anthropic" in module_name:
                return "anthropic"
            if "google" in module_name or "genai" in module_name:
                return "google"

    return None


def normalize_response(response: Any, provider: str) -> Tuple[str, str, Dict[str, int], Optional[str]]:
    """
    Normalize provider-specific response to common format.

    Args:
        response: Provider-specific response object
        provider: Provider name ("openai", "anthropic", "google")

    Returns:
        Tuple of (content, model, usage, finish_reason)
        - content: Response text
        - model: Model identifier
        - usage: Dict with prompt_tokens, completion_tokens, total_tokens
        - finish_reason: Finish reason string or None
    """
    if provider == "openai":
        return _normalize_openai_response(response)
    elif provider == "anthropic":
        return _normalize_anthropic_response(response)
    elif provider == "google":
        return _normalize_google_response(response)
    else:
        # Fallback: try to extract common fields
        return _normalize_generic_response(response)


def _normalize_openai_response(response: Any) -> Tuple[str, str, Dict[str, int], Optional[str]]:
    """Normalize OpenAI ChatCompletion response."""
    try:
        # OpenAI response structure
        choice = response.choices[0]
        content = choice.message.content or ""
        model = response.model
        finish_reason = choice.finish_reason

        # Usage information
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
            # OpenAI also provides cached tokens in some cases
            if hasattr(response.usage, "prompt_tokens_details"):
                details = response.usage.prompt_tokens_details
                if hasattr(details, "cached_tokens"):
                    usage["cached_tokens"] = details.cached_tokens

        return content, model, usage, finish_reason

    except (AttributeError, IndexError) as e:
        logger.warning(f"Failed to normalize OpenAI response: {e}")
        return "", "unknown", {}, None


def _normalize_anthropic_response(response: Any) -> Tuple[str, str, Dict[str, int], Optional[str]]:
    """Normalize Anthropic Message response."""
    try:
        # Anthropic response structure
        content = ""
        if hasattr(response, "content") and response.content:
            # Content can be list of content blocks
            if isinstance(response.content, list):
                # Extract text from all text blocks
                content = "".join(
                    block.text for block in response.content
                    if hasattr(block, "text")
                )
            else:
                content = str(response.content)

        model = getattr(response, "model", "unknown")
        finish_reason = getattr(response, "stop_reason", None)

        # Usage information
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                "completion_tokens": getattr(response.usage, "output_tokens", 0),
                "total_tokens": (
                    getattr(response.usage, "input_tokens", 0) +
                    getattr(response.usage, "output_tokens", 0)
                ),
            }
            # Anthropic provides cache metrics
            if hasattr(response.usage, "cache_creation_input_tokens"):
                usage["cache_creation_tokens"] = response.usage.cache_creation_input_tokens
            if hasattr(response.usage, "cache_read_input_tokens"):
                usage["cached_tokens"] = response.usage.cache_read_input_tokens

        return content, model, usage, finish_reason

    except (AttributeError, IndexError) as e:
        logger.warning(f"Failed to normalize Anthropic response: {e}")
        return "", "unknown", {}, None


def _normalize_google_response(response: Any) -> Tuple[str, str, Dict[str, int], Optional[str]]:
    """Normalize Google Gemini response."""
    try:
        # Google Gemini response structure
        content = ""
        if hasattr(response, "text"):
            content = response.text
        elif hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                parts = candidate.content.parts
                content = "".join(part.text for part in parts if hasattr(part, "text"))

        # Model extraction (may not be in response)
        model = "gemini-unknown"
        if hasattr(response, "model_version"):
            model = response.model_version

        # Finish reason
        finish_reason = None
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason"):
                finish_reason = str(candidate.finish_reason)

        # Usage information
        usage = {}
        if hasattr(response, "usage_metadata"):
            metadata = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(metadata, "candidates_token_count", 0),
                "total_tokens": getattr(metadata, "total_token_count", 0),
            }
            if hasattr(metadata, "cached_content_token_count"):
                usage["cached_tokens"] = metadata.cached_content_token_count

        return content, model, usage, finish_reason

    except (AttributeError, IndexError) as e:
        logger.warning(f"Failed to normalize Google response: {e}")
        return "", "unknown", {}, None


def _normalize_generic_response(response: Any) -> Tuple[str, str, Dict[str, int], Optional[str]]:
    """Fallback normalization for unknown response types."""
    content = ""
    model = "unknown"
    usage = {}
    finish_reason = None

    # Try common attribute names
    if hasattr(response, "content"):
        content = str(response.content)
    elif hasattr(response, "text"):
        content = str(response.text)
    elif hasattr(response, "message"):
        content = str(response.message)

    if hasattr(response, "model"):
        model = response.model

    # Try to extract usage
    if hasattr(response, "usage"):
        usage_obj = response.usage
        usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
            "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
            "total_tokens": getattr(usage_obj, "total_tokens", 0),
        }

    if hasattr(response, "finish_reason"):
        finish_reason = response.finish_reason

    return content, model, usage, finish_reason
