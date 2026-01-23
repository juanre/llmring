"""Alias resolution service with fallback support for model pools. Resolves aliases to available models based on configured API keys."""

import logging
import os
import threading
from typing import Dict, Optional, Set, Tuple

from cachetools import TTLCache

from llmring.lockfile_core import Lockfile, discover_package_lockfile
from llmring.utils import is_model_reference, parse_model_string

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
        # Cache for extended lockfiles (package_name -> Lockfile)
        self._extended_lockfiles: Dict[str, Optional[Lockfile]] = {}
        self._extended_lockfiles_lock = threading.Lock()

    def _parse_namespaced_alias(self, alias_or_model: str) -> Optional[Tuple[str, str]]:
        """
        Parse a namespaced alias (e.g., 'libA:summarizer').

        Distinguishes between:
        - Model references: 'openai:gpt-4' (known provider prefix)
        - Namespaced aliases: 'libA:summarizer' (package prefix from extends)

        Args:
            alias_or_model: Input string to parse

        Returns:
            Tuple of (namespace, alias_name) if namespaced alias, None otherwise
        """
        if ":" not in alias_or_model:
            return None

        # If it's a model reference (known provider prefix), not a namespaced alias
        if is_model_reference(alias_or_model):
            return None

        prefix, suffix = alias_or_model.split(":", 1)

        # Check if the prefix is in extends.packages (if we have a lockfile)
        if self.lockfile and hasattr(self.lockfile, "extends"):
            if prefix in self.lockfile.extends.packages:
                return (prefix, suffix)

        # If we don't have lockfile or prefix not in extends, still return as parsed
        # The caller will handle the "not in extends" error
        return (prefix, suffix)

    def _load_extended_lockfile(self, package_name: str) -> Optional[Lockfile]:
        """
        Load and cache a lockfile from an extended package.

        Thread-safe with double-checked locking for performance.

        Args:
            package_name: Name of the package to load lockfile from

        Returns:
            The loaded Lockfile, or None if not found
        """
        # First check (without lock) for cached value
        if package_name in self._extended_lockfiles:
            return self._extended_lockfiles[package_name]

        # Load lockfile outside lock to avoid blocking other threads
        lockfile_path = discover_package_lockfile(package_name)
        lockfile: Optional[Lockfile] = None
        if lockfile_path:
            try:
                lockfile = Lockfile.load(lockfile_path)
                logger.debug(f"Loaded lockfile from package '{package_name}': {lockfile_path}")
            except Exception as e:
                logger.warning(f"Failed to load lockfile from package '{package_name}': {e}")

        # Second check (with lock) before writing to cache
        with self._extended_lockfiles_lock:
            if package_name not in self._extended_lockfiles:
                self._extended_lockfiles[package_name] = lockfile
            return self._extended_lockfiles[package_name]

    def resolve(self, alias_or_model: str, profile: Optional[str] = None) -> str:
        """
        Resolve an alias to a model string, or return the input if it's already a model.

        Supports namespaced aliases (e.g., 'libA:summarizer') which resolve from
        extended packages defined in [extends].packages.

        Args:
            alias_or_model: Either an alias, namespaced alias, or model string (provider:model)
            profile: Optional profile name (defaults to lockfile default or env var)

        Returns:
            Resolved model string (provider:model) - first available from fallback list

        Raises:
            ValueError: If alias cannot be resolved or format is invalid
        """
        # Normalize profile early for consistent cache keys
        resolved_profile = profile or os.getenv("LLMRING_PROFILE")

        # Check for namespaced alias first (e.g., libA:summarizer)
        parsed = self._parse_namespaced_alias(alias_or_model)
        if parsed:
            namespace, alias_name = parsed
            if not namespace or not alias_name:
                raise ValueError(
                    f"Invalid namespaced alias: '{alias_or_model}'. "
                    f"Format must be 'package:alias' with non-empty components."
                )
            return self._resolve_namespaced_alias(namespace, alias_name, resolved_profile)

        # If it looks like a model reference (contains colon and is a known provider), return as-is
        if ":" in alias_or_model:
            return alias_or_model

        # Check cache first
        cache_key = (alias_or_model, resolved_profile)
        if cache_key in self._cache:
            cached_value = self._cache[cache_key]
            logger.debug(f"Using cached resolution for alias '{alias_or_model}': '{cached_value}'")
            return cached_value

        # Try to resolve from lockfile
        resolved = self._resolve_from_lockfile(alias_or_model, resolved_profile)
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

    def _resolve_namespaced_alias(
        self, namespace: str, alias_name: str, profile: Optional[str] = None
    ) -> str:
        """
        Resolve a namespaced alias (e.g., libA:summarizer).

        Resolution order:
        1. Check consumer lockfile for exact match (override)
        2. Load namespace package's lockfile and resolve alias

        Args:
            namespace: Package name (e.g., 'libA')
            alias_name: Alias within the package (e.g., 'summarizer')
            profile: Profile name (already resolved by caller)

        Returns:
            Resolved model string

        Raises:
            ValueError: If alias cannot be resolved
        """
        full_alias = f"{namespace}:{alias_name}"

        # Check cache first (profile is already resolved by caller)
        cache_key = (full_alias, profile)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 1. Check if consumer lockfile has an override
        if self.lockfile:
            model_refs = self.lockfile.resolve_alias(full_alias, profile)
            if model_refs:
                resolved = self._resolve_model_refs_to_first_available(model_refs, full_alias)
                if resolved:
                    self._cache[cache_key] = resolved
                    return resolved

        # 2. Check if namespace is in extends.packages
        if not self.lockfile or not hasattr(self.lockfile, "extends"):
            raise ValueError(
                f"Cannot resolve '{full_alias}'. "
                f"No lockfile configured or lockfile has no [extends] section."
            )

        if namespace not in self.lockfile.extends.packages:
            raise ValueError(
                f"Cannot resolve '{full_alias}'. "
                f"Package '{namespace}' is not listed in [extends].packages. "
                f"Add it to your lockfile: [extends] packages = [\"{namespace}\"]"
            )

        # 3. Load the package's lockfile
        package_lockfile = self._load_extended_lockfile(namespace)
        if not package_lockfile:
            raise ValueError(
                f"Cannot resolve '{full_alias}'. "
                f"Package '{namespace}' is in [extends] but has no llmring.lock. "
                f"Ensure the package includes a llmring.lock file."
            )

        # 4. Resolve from package lockfile
        model_refs = package_lockfile.resolve_alias(alias_name, profile)
        if not model_refs:
            available = package_lockfile.list_aliases(profile)
            available_str = ", ".join(available) if available else "(none)"
            raise ValueError(
                f"Alias '{full_alias}' not found. "
                f"Package '{namespace}' is in [extends] but its lockfile has no '{alias_name}' alias. "
                f"Available aliases in {namespace}: {available_str}"
            )

        resolved = self._resolve_model_refs_to_first_available(model_refs, full_alias)
        if resolved:
            self._cache[cache_key] = resolved
            return resolved

        raise ValueError(
            f"No available providers for alias '{full_alias}'. "
            f"Please configure the required API keys."
        )

    def _resolve_model_refs_to_first_available(
        self, model_refs: list, alias: str
    ) -> Optional[str]:
        """
        Resolve a list of model refs to the first available one.

        Args:
            model_refs: List of model references (provider:model)
            alias: Original alias (for logging)

        Returns:
            First available model reference, or None if none available
        """
        for model_ref in model_refs:
            try:
                provider_type, _ = self._parse_model_string(model_ref)
                if provider_type in self.available_providers:
                    logger.debug(f"Resolved alias '{alias}' to '{model_ref}'")
                    return model_ref
            except ValueError:
                logger.warning(f"Invalid model reference in alias '{alias}': {model_ref}")
                continue
        return None

    def resolve_candidates(self, alias_or_model: str, profile: Optional[str] = None) -> list[str]:
        """
        Resolve an alias to an ordered list of candidate model strings.

        Args:
            alias_or_model: Either an alias or a model string (provider:model)
            profile: Optional profile name (defaults to lockfile default or env var)

        Returns:
            Ordered list of resolved model strings (provider:model), filtered to configured providers

        Raises:
            ValueError: If alias cannot be resolved, has invalid entries, or no providers are available
        """
        if ":" in alias_or_model:
            return [alias_or_model]

        if not self.lockfile:
            raise ValueError(
                f"Invalid model format: '{alias_or_model}'. "
                f"Models must be specified as 'provider:model' (e.g., 'openai:gpt-4'). "
                f"If you meant to use an alias, ensure it's defined in your lockfile."
            )

        profile = profile or os.getenv("LLMRING_PROFILE")
        model_refs = self.lockfile.resolve_alias(alias_or_model, profile)
        if not model_refs:
            raise ValueError(
                f"Invalid model format: '{alias_or_model}'. "
                f"Models must be specified as 'provider:model' (e.g., 'openai:gpt-4'). "
                f"If you meant to use an alias, ensure it's defined in your lockfile."
            )

        candidates: list[str] = []
        unavailable_models: list[str] = []
        for model_ref in model_refs:
            try:
                provider_type, _ = self._parse_model_string(model_ref)
            except ValueError:
                logger.warning(
                    "Invalid model reference in alias '%s': %s", alias_or_model, model_ref
                )
                continue

            if provider_type in self.available_providers:
                candidates.append(model_ref)
            else:
                unavailable_models.append(f"{model_ref} (no {provider_type} API key)")

        if not candidates:
            raise ValueError(
                f"No available providers for alias '{alias_or_model}'. "
                f"Tried models: {', '.join(unavailable_models)}. "
                f"Please configure the required API keys."
            )

        return candidates

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
        profile = profile or os.getenv("LLMRING_PROFILE")
        model_refs = self.lockfile.resolve_alias(alias, profile)

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
