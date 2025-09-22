"""
Registry-aware model selection for tests.

This module provides valid model names that are known to be in the registry,
so tests can reliably use them without hardcoding models that might not exist.
"""

# Models that are known to be in the registry (as of the last check)
# These should be updated periodically to match registry contents

VALID_ANTHROPIC_MODELS = [
    "claude-3-opus-20240229",      # Stable, widely available
    "claude-3-haiku-20240307",     # Fast, affordable
    "claude-3-5-haiku-20241022",   # Newer haiku version with vision
    "claude-opus-4-20250514",      # Latest opus with vision
    "claude-sonnet-4-20250514",    # Latest sonnet with vision
]

VALID_OPENAI_MODELS = [
    "gpt-4o-mini",                 # Has vision support
    "gpt-4",                       # Standard GPT-4
    "gpt-3.5-turbo",              # Fast, affordable
]

VALID_GOOGLE_MODELS = [
    "gemini-1.5-flash",            # Fast, affordable with vision
    "gemini-1.5-pro",              # More capable
    "gemini-2.0-flash",            # Newer version with vision
]

# Default models to use in tests (should be stable and available)
DEFAULT_TEST_MODELS = {
    "anthropic": "claude-3-haiku-20240307",  # Fast and cheap for tests
    "openai": "gpt-4o-mini",                 # Multimodal and in registry
    "google": "gemini-1.5-flash",            # Fast Gemini model
}

# Models with specific capabilities
VISION_CAPABLE_MODELS = {
    "anthropic": "claude-3-5-haiku-20241022",  # Has vision support
    "openai": "gpt-4o-mini",                   # GPT-4o-mini has vision
    "google": "gemini-1.5-flash",              # All Gemini models support vision
}

PDF_CAPABLE_MODELS = {
    "anthropic": "claude-3-5-haiku-20241022",  # Newer model with PDF support
    "google": "gemini-1.5-flash",              # Can process documents
    "openai": "gpt-4o-mini",                   # Via assistants API
}

def get_test_model(provider: str, capability: str = None) -> str:
    """
    Get a valid test model for a given provider and optional capability.

    Args:
        provider: Provider name (anthropic, openai, google)
        capability: Optional capability (vision, pdf, etc.)

    Returns:
        Model name that should be in the registry
    """
    if capability == "vision":
        return VISION_CAPABLE_MODELS.get(provider, DEFAULT_TEST_MODELS.get(provider))
    elif capability == "pdf":
        return PDF_CAPABLE_MODELS.get(provider, DEFAULT_TEST_MODELS.get(provider))
    else:
        return DEFAULT_TEST_MODELS.get(provider)

def get_all_valid_models(provider: str) -> list:
    """Get all valid models for a provider."""
    models_map = {
        "anthropic": VALID_ANTHROPIC_MODELS,
        "openai": VALID_OPENAI_MODELS,
        "google": VALID_GOOGLE_MODELS,
    }
    return models_map.get(provider, [])