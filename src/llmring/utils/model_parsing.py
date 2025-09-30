"""
Utilities for parsing model reference strings.

This module provides a centralized function for parsing the 'provider:model' format
used throughout LLMRing, eliminating duplicate parsing logic.
"""


def parse_model_string(model: str) -> tuple[str, str]:
    """
    Parse a model string into provider and model name.

    Args:
        model: Must be in provider:model format (e.g., "anthropic:claude-3-opus")

    Returns:
        Tuple of (provider_type, model_name)

    Raises:
        ValueError: If model string is not in provider:model format

    Examples:
        >>> parse_model_string("anthropic:claude-3-opus-20240229")
        ('anthropic', 'claude-3-opus-20240229')

        >>> parse_model_string("openai:gpt-4")
        ('openai', 'gpt-4')

        >>> parse_model_string("invalid")  # doctest: +SKIP
        ValueError: Invalid model format: 'invalid'. Models must be specified as 'provider:model'
    """
    if ":" not in model:
        raise ValueError(
            f"Invalid model format: '{model}'. "
            f"Models must be specified as 'provider:model' (e.g., 'openai:gpt-4'). "
            f"If you meant to use an alias, ensure it's defined in your lockfile."
        )

    provider_type, model_name = model.split(":", 1)

    if not provider_type or not model_name:
        raise ValueError(
            f"Invalid model format: '{model}'. " f"Both provider and model name must be non-empty."
        )

    return provider_type, model_name


def strip_provider_prefix(model: str, provider: str) -> str:
    """
    Strip provider prefix from model string if present.

    Args:
        model: Model string, possibly with provider prefix (e.g., "anthropic:claude-3-opus")
        provider: Expected provider name (e.g., "anthropic")

    Returns:
        Model name without provider prefix (e.g., "claude-3-opus")

    Examples:
        >>> strip_provider_prefix("anthropic:claude-3-opus", "anthropic")
        'claude-3-opus'

        >>> strip_provider_prefix("claude-3-opus", "anthropic")
        'claude-3-opus'

        >>> strip_provider_prefix("openai:gpt-4", "anthropic")
        'openai:gpt-4'
    """
    prefix = f"{provider.lower()}:"
    if model.lower().startswith(prefix):
        return model.split(":", 1)[1]
    return model
