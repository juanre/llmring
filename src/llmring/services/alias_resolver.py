"""
Alias resolution service for LLMRing.

Handles resolving model aliases to concrete provider:model references,
with caching and fallback support.
"""

import logging
import os
from typing import Dict, Optional, Set

from cachetools import TTLCache

from llmring.lockfile_core import Lockfile
from llmring.utils import parse_model_string

logger = logging.getLogger(__name__)


class AliasResolver:
    """
    Resolves model aliases to concrete provider:model references.

    The resolver checks:
    1. If input is already provider:model format, returns as-is
    2. Checks cache for previously resolved aliases
    3. Resolves from lockfile with fallback support
    4. Returns first available provider from fallback list
    """

    def __init__(
        self,
        lockfile: Optional[Lockfile] = None,
        available_providers: Optional[Set[str]] = None,
        cache_size: int = 100,
        cache_ttl: int = 3600,
    ):
        """
        Initialize the alias resolver.

        Args:
            lockfile: Lockfile containing alias definitions
            available_providers: Set of provider names with configured API keys
            cache_size: Maximum number of cached alias resolutions
            cache_ttl: TTL for cache entries in seconds
        """
        self.lockfile = lockfile
        self.available_providers = available_providers or set()
        self._cache: TTLCache = TTLCache(maxsize=cache_size, ttl=cache_ttl)

    def resolve(self, alias_or_model: str, profile: Optional[str] = None) -> str:
        """
        Resolve an alias to a model string, or return the input if it's already a model.

        Args:
            alias_or_model: Either an alias or a model string (provider:model)
            profile: Optional profile name (defaults to lockfile default or env var)

        Returns:
            Resolved model string (provider:model) - first available from fallback list

        Raises:
            ValueError: If alias cannot be resolved or format is invalid
        """
        # If it looks like a model reference (contains colon), return as-is
        if ":" in alias_or_model:
            return alias_or_model

        # Check cache first
        cache_key = (alias_or_model, profile)
        if cache_key in self._cache:
            cached_value = self._cache[cache_key]
            logger.debug(f"Using cached resolution for alias '{alias_or_model}': '{cached_value}'")
            return cached_value

        # Try to resolve from lockfile
        resolved = self._resolve_from_lockfile(alias_or_model, profile)
        if resolved:
            # Add to cache
            self._cache[cache_key] = resolved
            return resolved

        # If no lockfile or alias not found, this is an error
        raise ValueError(
            f"Invalid model format: '{alias_or_model}'. "
            f"Models must be specified as 'provider:model' (e.g., 'openai:gpt-4'). "
            f"If you meant to use an alias, ensure it's defined in your lockfile."
        )

    def _resolve_from_lockfile(self, alias: str, profile: Optional[str] = None) -> Optional[str]:
        """
        Resolve alias from lockfile, checking for available providers.

        Args:
            alias: The alias to resolve
            profile: Optional profile name

        Returns:
            First available model from fallback list, or None if not found
        """
        if not self.lockfile:
            return None

        # Get profile name from argument, environment, or lockfile default
        profile_name = profile or os.getenv("LLMRING_PROFILE")
        model_refs = self.lockfile.resolve_alias(alias, profile_name)

        if not model_refs:
            return None

        # Try each model in order until we find one with an available provider
        unavailable_models = []
        for model_ref in model_refs:
            try:
                provider_type, _ = self._parse_model_string(model_ref)
                if provider_type in self.available_providers:
                    logger.debug(f"Resolved alias '{alias}' to '{model_ref}' (provider available)")
                    return model_ref
                else:
                    unavailable_models.append(f"{model_ref} (no {provider_type} API key)")
                    logger.debug(
                        f"Skipping '{model_ref}' - provider '{provider_type}' not available"
                    )
            except ValueError:
                # Invalid model reference format
                logger.warning(f"Invalid model reference in alias '{alias}': {model_ref}")
                continue

        # No available providers found
        if unavailable_models:
            raise ValueError(
                f"No available providers for alias '{alias}'. "
                f"Tried models: {', '.join(unavailable_models)}. "
                f"Please configure the required API keys."
            )

        return None

    @staticmethod
    def _parse_model_string(model: str) -> tuple[str, str]:
        """
        Parse a model string into provider and model name.

        Args:
            model: Model string in format 'provider:model'

        Returns:
            Tuple of (provider_type, model_name)

        Raises:
            ValueError: If model string format is invalid
        """
        if ":" not in model:
            raise ValueError(f"Model must be in format 'provider:model', got: {model}")
        provider_type, model_name = parse_model_string(model)
        return provider_type, model_name

    def clear_cache(self):
        """Clear the alias resolution cache."""
        self._cache.clear()
        logger.debug("Alias cache cleared")

    def update_available_providers(self, providers: Set[str]):
        """
        Update the set of available providers.

        Args:
            providers: Set of provider names with configured API keys
        """
        self.available_providers = providers
        # Clear cache since availability has changed
        self.clear_cache()
        logger.debug(f"Updated available providers: {providers}")
