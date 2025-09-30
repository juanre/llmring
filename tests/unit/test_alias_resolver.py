"""Unit tests for AliasResolver service."""

import os
from unittest.mock import MagicMock

import pytest

from llmring.lockfile_core import Lockfile
from llmring.services.alias_resolver import AliasResolver


class TestAliasResolver:
    """Tests for AliasResolver service."""

    def test_resolve_direct_model_reference(self):
        """Should return provider:model format as-is."""
        resolver = AliasResolver()
        result = resolver.resolve("openai:gpt-4")
        assert result == "openai:gpt-4"

    def test_resolve_with_cache(self):
        """Should use cached values."""
        lockfile = MagicMock(spec=Lockfile)
        lockfile.resolve_alias.return_value = ["openai:gpt-4"]

        resolver = AliasResolver(lockfile=lockfile, available_providers={"openai"})

        # First call - should hit lockfile
        result1 = resolver.resolve("fast")
        assert result1 == "openai:gpt-4"
        assert lockfile.resolve_alias.call_count == 1

        # Second call - should use cache
        result2 = resolver.resolve("fast")
        assert result2 == "openai:gpt-4"
        assert lockfile.resolve_alias.call_count == 1  # Not called again

    def test_resolve_with_fallback(self):
        """Should try fallback models in order."""
        lockfile = MagicMock(spec=Lockfile)
        lockfile.resolve_alias.return_value = [
            "anthropic:claude-3-5-sonnet-20241022",
            "openai:gpt-4",
        ]

        resolver = AliasResolver(
            lockfile=lockfile, available_providers={"openai"}  # Only OpenAI available
        )

        result = resolver.resolve("fast")
        assert result == "openai:gpt-4"  # Should skip Anthropic, use OpenAI

    def test_resolve_no_available_provider(self):
        """Should raise error when no provider available."""
        lockfile = MagicMock(spec=Lockfile)
        lockfile.resolve_alias.return_value = [
            "anthropic:claude-3-5-sonnet-20241022",
            "openai:gpt-4",
        ]

        resolver = AliasResolver(
            lockfile=lockfile, available_providers=set()  # No providers available
        )

        with pytest.raises(ValueError, match="No available providers"):
            resolver.resolve("fast")

    def test_resolve_alias_not_found(self):
        """Should raise error when alias not in lockfile."""
        lockfile = MagicMock(spec=Lockfile)
        lockfile.resolve_alias.return_value = []

        resolver = AliasResolver(lockfile=lockfile, available_providers={"openai"})

        with pytest.raises(ValueError, match="Invalid model format"):
            resolver.resolve("nonexistent")

    def test_resolve_no_lockfile(self):
        """Should raise error when no lockfile and not provider:model format."""
        resolver = AliasResolver(lockfile=None, available_providers={"openai"})

        with pytest.raises(ValueError, match="Invalid model format"):
            resolver.resolve("fast")

    def test_resolve_with_profile(self):
        """Should pass profile to lockfile."""
        lockfile = MagicMock(spec=Lockfile)
        lockfile.resolve_alias.return_value = ["openai:gpt-4"]

        resolver = AliasResolver(lockfile=lockfile, available_providers={"openai"})

        resolver.resolve("fast", profile="production")
        lockfile.resolve_alias.assert_called_with("fast", "production")

    def test_resolve_with_env_profile(self, monkeypatch):
        """Should use LLMRING_PROFILE env var if no profile argument."""
        monkeypatch.setenv("LLMRING_PROFILE", "staging")

        lockfile = MagicMock(spec=Lockfile)
        lockfile.resolve_alias.return_value = ["openai:gpt-4"]

        resolver = AliasResolver(lockfile=lockfile, available_providers={"openai"})

        resolver.resolve("fast")
        lockfile.resolve_alias.assert_called_with("fast", "staging")

    def test_parse_model_string_valid(self):
        """Should parse valid provider:model format."""
        provider, model = AliasResolver._parse_model_string("openai:gpt-4")
        assert provider == "openai"
        assert model == "gpt-4"

    def test_parse_model_string_with_colons(self):
        """Should handle model names with colons."""
        provider, model = AliasResolver._parse_model_string("openai:gpt-4:turbo")
        assert provider == "openai"
        assert model == "gpt-4:turbo"

    def test_parse_model_string_invalid(self):
        """Should raise error for invalid format."""
        with pytest.raises(ValueError, match="Model must be in format"):
            AliasResolver._parse_model_string("gpt-4")

    def test_clear_cache(self):
        """Should clear the cache."""
        lockfile = MagicMock(spec=Lockfile)
        lockfile.resolve_alias.return_value = ["openai:gpt-4"]

        resolver = AliasResolver(lockfile=lockfile, available_providers={"openai"})

        # Populate cache
        resolver.resolve("fast")
        assert lockfile.resolve_alias.call_count == 1

        # Clear cache
        resolver.clear_cache()

        # Should hit lockfile again
        resolver.resolve("fast")
        assert lockfile.resolve_alias.call_count == 2

    def test_update_available_providers(self):
        """Should update providers and clear cache."""
        lockfile = MagicMock(spec=Lockfile)
        lockfile.resolve_alias.return_value = ["openai:gpt-4"]

        resolver = AliasResolver(lockfile=lockfile, available_providers={"openai"})

        # Populate cache
        resolver.resolve("fast")

        # Update providers
        resolver.update_available_providers({"anthropic", "openai"})

        # Cache should be cleared, so lockfile called again
        resolver.resolve("fast")
        assert lockfile.resolve_alias.call_count == 2

    def test_cache_ttl_expiry(self):
        """Should not use cache after TTL expires."""
        import time

        lockfile = MagicMock(spec=Lockfile)
        lockfile.resolve_alias.return_value = ["openai:gpt-4"]

        resolver = AliasResolver(
            lockfile=lockfile, available_providers={"openai"}, cache_ttl=1  # 1 second TTL
        )

        # First call
        resolver.resolve("fast")
        assert lockfile.resolve_alias.call_count == 1

        # Immediate second call - should use cache
        resolver.resolve("fast")
        assert lockfile.resolve_alias.call_count == 1

        # Wait for TTL to expire
        time.sleep(1.1)

        # Should call lockfile again
        resolver.resolve("fast")
        assert lockfile.resolve_alias.call_count == 2

    def test_invalid_model_reference_in_lockfile(self):
        """Should skip invalid model references and continue."""
        lockfile = MagicMock(spec=Lockfile)
        lockfile.resolve_alias.return_value = [
            "invalid-format",  # No colon
            "openai:gpt-4",  # Valid
        ]

        resolver = AliasResolver(lockfile=lockfile, available_providers={"openai"})

        # Should skip invalid and use valid one
        result = resolver.resolve("fast")
        assert result == "openai:gpt-4"
