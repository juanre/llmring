"""
Token counting utilities for LLMRing.

Provides accurate token counting for different providers using their respective tokenizers.
Falls back to tiktoken for OpenAI-compatible models and character-based estimation as last resort.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Cache for tokenizers to avoid repeated initialization
_tokenizer_cache: Dict[str, Any] = {}


def count_tokens_openai(messages: List[Dict[str, Any]], model: str) -> int:
    """
    Count tokens for OpenAI models using tiktoken.

    Args:
        messages: List of message dictionaries
        model: Model name for encoding selection

    Returns:
        Token count
    """
    try:
        import tiktoken
    except ImportError:
        logger.warning("tiktoken not installed, using character estimation")
        return _estimate_tokens_from_messages(messages)

    # Get the correct encoding for the model
    try:
        if "gpt-4o" in model:
            encoding = tiktoken.get_encoding("o200k_base")
        elif "gpt-4" in model:
            encoding = tiktoken.get_encoding("cl100k_base")
        elif "gpt-3.5" in model:
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")  # Default
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Cache the encoding
    _tokenizer_cache[f"openai_{model}"] = encoding

    # Count tokens in messages
    num_tokens = 0
    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4  # <im_start>, role/name, \n, <im_end>\n

        # Add role tokens
        role = message.get("role", "")
        num_tokens += len(encoding.encode(role))

        # Add content tokens
        content = message.get("content", "")
        if isinstance(content, str):
            num_tokens += len(encoding.encode(content))
        elif isinstance(content, list):
            # Handle multimodal content
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text = part.get("text", "")
                        num_tokens += len(encoding.encode(text))
                    elif part.get("type") == "image_url":
                        # Images have a base token cost
                        # This is a rough estimate; actual cost depends on image size
                        num_tokens += 85  # Base cost for an image

        # Add name if present
        if "name" in message:
            num_tokens += len(encoding.encode(message["name"]))
            num_tokens -= 1  # Role is omitted if name is present

    num_tokens += 2  # Every reply is primed with <im_start>assistant
    return num_tokens


def count_tokens_anthropic(messages: List[Dict[str, Any]], model: str) -> int:
    """
    Count tokens for Anthropic models.

    Note: Anthropic doesn't provide a public tokenizer, so we use estimation.
    Their token count is generally similar to OpenAI's cl100k_base encoding.

    Args:
        messages: List of message dictionaries
        model: Model name

    Returns:
        Estimated token count
    """
    try:
        import tiktoken

        # Use cl100k_base as approximation for Claude
        encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                num_tokens += len(encoding.encode(content))
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        num_tokens += len(encoding.encode(text))

        # Add some overhead for message structure
        num_tokens += len(messages) * 4
        return num_tokens

    except ImportError:
        return _estimate_tokens_from_messages(messages)


def count_tokens_google(messages: List[Dict[str, Any]], model: str) -> int:
    """
    Count tokens for Google models.

    Google uses a different tokenization approach. This is an estimation.

    Args:
        messages: List of message dictionaries
        model: Model name

    Returns:
        Estimated token count
    """
    # Google's tokenization is generally similar to OpenAI's
    # but can vary. Using character estimation for now.
    return _estimate_tokens_from_messages(messages)


def _estimate_tokens_from_messages(messages: List[Dict[str, Any]]) -> int:
    """
    Fallback token estimation based on character count.

    Uses the rough approximation of 1 token per 4 characters.

    Args:
        messages: List of message dictionaries

    Returns:
        Estimated token count
    """
    total_chars = 0

    for message in messages:
        # Count role
        total_chars += len(message.get("role", ""))

        # Count content
        content = message.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        total_chars += len(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        # Rough estimate for image tokens
                        total_chars += 340  # ~85 tokens * 4 chars

    # Rough estimate: 1 token per 4 characters
    return total_chars // 4


def count_tokens(messages: List[Dict[str, Any]], provider: str, model: str) -> int:
    """
    Count tokens for a list of messages based on provider and model.

    Args:
        messages: List of message dictionaries
        provider: Provider name (openai, anthropic, google, ollama)
        model: Model name

    Returns:
        Token count (exact for OpenAI, estimated for others)
    """
    if provider == "openai":
        return count_tokens_openai(messages, model)
    elif provider == "anthropic":
        return count_tokens_anthropic(messages, model)
    elif provider == "google":
        return count_tokens_google(messages, model)
    elif provider == "ollama":
        # Ollama models vary widely, use character estimation
        return _estimate_tokens_from_messages(messages)
    else:
        # Unknown provider, use fallback
        return _estimate_tokens_from_messages(messages)


def count_string_tokens(text: str, provider: str, model: str) -> int:
    """
    Count tokens in a simple string.

    Args:
        text: Text to tokenize
        provider: Provider name
        model: Model name

    Returns:
        Token count
    """
    # Convert to message format for consistency
    messages = [{"role": "user", "content": text}]

    # Subtract the overhead added for message structure
    base_count = count_tokens(messages, provider, model)

    # Remove the message structure overhead (roughly 4-6 tokens)
    return max(0, base_count - 6)
