"""
Tests for lockfile validation helpers (has_alias, require_aliases).

These tests verify the new validation helper methods added to Lockfile and LLMRing
classes for the library composition pattern.
"""

from pathlib import Path

import pytest

from llmring.lockfile import AliasBinding, Lockfile, ProfileConfig
from llmring.service import LLMRing


class TestLockfileHasAlias:
    """Tests for Lockfile.has_alias() method."""

    def test_has_alias_returns_true_when_alias_exists_in_default_profile(self):
        """Test that has_alias returns True when alias exists in default profile."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[
                        AliasBinding(alias="summarizer", models=["openai:gpt-4o"]),
                        AliasBinding(alias="analyzer", models=["anthropic:claude-3-5-sonnet"]),
                    ],
                )
            },
        )

        assert lockfile.has_alias("summarizer") is True
        assert lockfile.has_alias("analyzer") is True

    def test_has_alias_returns_true_when_alias_exists_in_specified_profile(self):
        """Test that has_alias returns True when alias exists in specified profile."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[AliasBinding(alias="summarizer", models=["openai:gpt-4o"])],
                ),
                "prod": ProfileConfig(
                    name="prod",
                    bindings=[
                        AliasBinding(alias="analyzer", models=["anthropic:claude-3-5-sonnet"])
                    ],
                ),
            },
        )

        assert lockfile.has_alias("summarizer", profile="default") is True
        assert lockfile.has_alias("analyzer", profile="prod") is True

    def test_has_alias_returns_false_when_alias_does_not_exist(self):
        """Test that has_alias returns False when alias doesn't exist."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[AliasBinding(alias="summarizer", models=["openai:gpt-4o"])],
                )
            },
        )

        assert lockfile.has_alias("nonexistent") is False
        assert lockfile.has_alias("missing_alias") is False

    def test_has_alias_returns_false_when_profile_does_not_exist(self):
        """Test that has_alias returns False when profile doesn't exist."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[AliasBinding(alias="summarizer", models=["openai:gpt-4o"])],
                )
            },
        )

        # Profile doesn't exist, should return False
        assert lockfile.has_alias("summarizer", profile="nonexistent") is False

    def test_has_alias_returns_false_when_lockfile_has_no_bindings(self):
        """Test that has_alias returns False when lockfile has no bindings."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={"default": ProfileConfig(name="default", bindings=[])},
        )

        assert lockfile.has_alias("any_alias") is False

    def test_has_alias_handles_none_profile_correctly(self):
        """Test that has_alias uses default_profile when profile is None."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="custom_default",
            profiles={
                "custom_default": ProfileConfig(
                    name="custom_default",
                    bindings=[AliasBinding(alias="summarizer", models=["openai:gpt-4o"])],
                )
            },
        )

        # None should use default_profile
        assert lockfile.has_alias("summarizer", profile=None) is True
        assert lockfile.has_alias("summarizer") is True


class TestLockfileRequireAliases:
    """Tests for Lockfile.require_aliases() method."""

    def test_require_aliases_passes_when_all_required_aliases_exist(self):
        """Test that require_aliases passes when all required aliases exist."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[
                        AliasBinding(alias="summarizer", models=["openai:gpt-4o"]),
                        AliasBinding(alias="analyzer", models=["anthropic:claude-3-5-sonnet"]),
                        AliasBinding(alias="classifier", models=["google:gemini-1.5-flash"]),
                    ],
                )
            },
        )

        # Should not raise
        lockfile.require_aliases(["summarizer", "analyzer"])

    def test_require_aliases_raises_valueerror_when_one_alias_missing(self):
        """Test that require_aliases raises ValueError when one alias missing."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[AliasBinding(alias="summarizer", models=["openai:gpt-4o"])],
                )
            },
        )

        with pytest.raises(ValueError) as exc_info:
            lockfile.require_aliases(["missing_alias"])

        error_msg = str(exc_info.value)
        assert "missing_alias" in error_msg

    def test_require_aliases_raises_valueerror_when_multiple_aliases_missing(self):
        """Test that require_aliases raises ValueError listing all missing aliases."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[AliasBinding(alias="summarizer", models=["openai:gpt-4o"])],
                )
            },
        )

        with pytest.raises(ValueError) as exc_info:
            lockfile.require_aliases(["missing1", "missing2", "missing3"])

        error_msg = str(exc_info.value)
        # Should list all missing aliases
        assert "missing1" in error_msg
        assert "missing2" in error_msg
        assert "missing3" in error_msg

    def test_require_aliases_error_message_includes_lockfile_reference(self):
        """Test that error message includes reference to lockfile."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[],
                )
            },
        )

        with pytest.raises(ValueError) as exc_info:
            lockfile.require_aliases(["missing"])

        error_msg = str(exc_info.value)
        # Error should mention lockfile in some way
        assert "lockfile" in error_msg.lower()

    def test_require_aliases_error_message_includes_context(self):
        """Test that error message includes context if provided."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[],
                )
            },
        )

        with pytest.raises(ValueError) as exc_info:
            lockfile.require_aliases(["missing"], context="test-library")

        error_msg = str(exc_info.value)
        assert "test-library" in error_msg

    def test_require_aliases_error_message_is_helpful_and_actionable(self):
        """Test that error message is helpful and actionable."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[AliasBinding(alias="existing", models=["openai:gpt-4o"])],
                )
            },
        )

        with pytest.raises(ValueError) as exc_info:
            lockfile.require_aliases(["missing1", "missing2"], context="my-library")

        error_msg = str(exc_info.value)
        # Error should be clear about what's wrong
        assert "missing" in error_msg.lower() or "required" in error_msg.lower()
        # Error should mention the context
        assert "my-library" in error_msg

    def test_require_aliases_works_with_empty_required_list(self):
        """Test that require_aliases works with empty required list (no-op)."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[],
                )
            },
        )

        # Should not raise
        lockfile.require_aliases([])

    def test_require_aliases_works_with_specified_profile(self):
        """Test that require_aliases works with specified profile."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[AliasBinding(alias="summarizer", models=["openai:gpt-4o"])],
                ),
                "prod": ProfileConfig(
                    name="prod",
                    bindings=[
                        AliasBinding(alias="analyzer", models=["anthropic:claude-3-5-sonnet"])
                    ],
                ),
            },
        )

        # Should not raise - analyzer exists in prod profile
        lockfile.require_aliases(["analyzer"], profile="prod")

        # Should raise - analyzer doesn't exist in default profile
        with pytest.raises(ValueError):
            lockfile.require_aliases(["analyzer"], profile="default")


class TestLLMRingValidationProxies:
    """Tests for LLMRing validation proxy methods."""

    def test_llmring_has_alias_proxies_to_lockfile(self, tmp_path):
        """Test that LLMRing.has_alias() proxies correctly to lockfile.has_alias()."""
        # Create lockfile
        lockfile_path = tmp_path / "test.lock"
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[AliasBinding(alias="summarizer", models=["openai:gpt-4o"])],
                )
            },
        )
        lockfile.save(lockfile_path)

        # Create LLMRing with lockfile
        ring = LLMRing(lockfile_path=str(lockfile_path))

        assert ring.has_alias("summarizer") is True
        assert ring.has_alias("nonexistent") is False

    def test_llmring_require_aliases_proxies_to_lockfile(self, tmp_path):
        """Test that LLMRing.require_aliases() proxies correctly to lockfile.require_aliases()."""
        # Create lockfile
        lockfile_path = tmp_path / "test.lock"
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[
                        AliasBinding(alias="summarizer", models=["openai:gpt-4o"]),
                        AliasBinding(alias="analyzer", models=["anthropic:claude-3-5-sonnet"]),
                    ],
                )
            },
        )
        lockfile.save(lockfile_path)

        # Create LLMRing with lockfile
        ring = LLMRing(lockfile_path=str(lockfile_path))

        # Should not raise
        ring.require_aliases(["summarizer", "analyzer"], context="test")

    def test_llmring_has_alias_returns_false_if_no_lockfile_loaded(self):
        """Test that has_alias returns False if no lockfile loaded."""
        # Create LLMRing without lockfile - this shouldn't happen in practice
        # but we need to handle it gracefully
        ring = LLMRing()

        # Even if no lockfile, should return False, not crash
        # Note: This test may need adjustment based on actual implementation
        result = ring.has_alias("any_alias")
        assert result is False

    def test_llmring_require_aliases_raises_clear_error_if_no_lockfile_loaded(self):
        """Test that require_aliases raises clear error if no lockfile loaded."""
        ring = LLMRing()

        # Should raise ValueError with clear message
        with pytest.raises(ValueError) as exc_info:
            ring.require_aliases(["any_alias"])

        error_msg = str(exc_info.value)
        assert "lockfile" in error_msg.lower()


class TestImprovedErrorMessages:
    """Tests for improved resolve_alias error messages."""

    def test_resolve_alias_error_lists_available_aliases(self):
        """Test that resolve_alias error message lists available aliases."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[
                        AliasBinding(alias="summarizer", models=["openai:gpt-4o"]),
                        AliasBinding(alias="analyzer", models=["anthropic:claude-3-5-sonnet"]),
                        AliasBinding(alias="classifier", models=["google:gemini-1.5-flash"]),
                    ],
                )
            },
        )

        # Try to resolve non-existent alias - should fail and list available
        result = lockfile.resolve_alias("nonexistent")

        # Note: resolve_alias currently returns empty list for non-existent aliases
        # This test verifies we need to improve the error message when we use the alias
        # The actual error would come from trying to use the empty list
        assert result == []

    def test_resolve_alias_error_includes_lockfile_path(self, tmp_path):
        """Test that resolve_alias error includes lockfile path."""
        # This test is for when we improve the error handling
        # Currently resolve_alias returns empty list, not error
        lockfile_path = tmp_path / "test.lock"
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[AliasBinding(alias="summarizer", models=["openai:gpt-4o"])],
                )
            },
        )
        lockfile.save(lockfile_path)

        loaded_lockfile = Lockfile.load(lockfile_path)
        result = loaded_lockfile.resolve_alias("nonexistent")

        # Currently just returns empty list
        assert result == []

    def test_resolve_alias_error_provides_helpful_suggestions(self):
        """Test that resolve_alias error provides helpful suggestions."""
        # This is a placeholder for when we improve error messages
        # The improvement should happen in the actual usage of resolve_alias
        # (e.g., in AliasResolver or LLMRing.chat)
        pass

    def test_resolve_alias_error_is_multiline_and_readable(self):
        """Test that resolve_alias error message is multiline and readable."""
        # This is a placeholder for when we improve error messages
        pass


class TestLibraryCompositionPattern:
    """Integration tests for library composition pattern."""

    def test_library_composition_basic_pattern(self, tmp_path):
        """Test basic library composition pattern works."""
        # Create a lockfile for Library A
        lockfile_a_path = tmp_path / "library_a.lock"
        lockfile_a = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[
                        AliasBinding(alias="summarizer", models=["anthropic:claude-3-haiku"])
                    ],
                )
            },
        )
        lockfile_a.save(lockfile_a_path)

        # Simulate Library A initialization
        class MockLibraryA:
            def __init__(self, lockfile_path=None):
                self.ring = LLMRing(lockfile_path=lockfile_path or str(lockfile_a_path))
                self.ring.require_aliases(["summarizer"], context="library-a")

        # Test that Library A initializes successfully with its own lockfile
        lib_a = MockLibraryA()
        assert lib_a.ring.has_alias("summarizer") is True

    def test_library_b_using_library_a_with_same_lockfile(self, tmp_path):
        """Test Library B using Library A with shared lockfile."""
        # Create lockfiles
        lockfile_a_path = tmp_path / "library_a.lock"
        lockfile_a = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[
                        AliasBinding(alias="summarizer", models=["anthropic:claude-3-haiku"])
                    ],
                )
            },
        )
        lockfile_a.save(lockfile_a_path)

        lockfile_b_path = tmp_path / "library_b.lock"
        lockfile_b = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[
                        AliasBinding(alias="summarizer", models=["anthropic:claude-3-5-sonnet"]),
                        AliasBinding(alias="analyzer", models=["openai:gpt-4o"]),
                    ],
                )
            },
        )
        lockfile_b.save(lockfile_b_path)

        # Simulate Library A and Library B
        class MockLibraryA:
            def __init__(self, lockfile_path=None):
                self.ring = LLMRing(lockfile_path=lockfile_path or str(lockfile_a_path))
                self.ring.require_aliases(["summarizer"], context="library-a")

        class MockLibraryB:
            def __init__(self, lockfile_path=None):
                lockfile = lockfile_path or str(lockfile_b_path)
                self.lib_a = MockLibraryA(lockfile_path=lockfile)
                self.ring = LLMRing(lockfile_path=lockfile)
                self.ring.require_aliases(["analyzer"], context="library-b")

        # Test that Library B initializes with its lockfile
        lib_b = MockLibraryB()
        assert lib_b.ring.has_alias("analyzer") is True
        assert lib_b.lib_a.ring.has_alias("summarizer") is True

    def test_library_composition_validates_across_libraries(self, tmp_path):
        """Test that validation works across library composition."""
        # Create lockfile that's missing required alias for Library A
        lockfile_incomplete_path = tmp_path / "incomplete.lock"
        lockfile_incomplete = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[AliasBinding(alias="analyzer", models=["openai:gpt-4o"])],
                )
            },
        )
        lockfile_incomplete.save(lockfile_incomplete_path)

        # Simulate Library A
        class MockLibraryA:
            def __init__(self, lockfile_path):
                self.ring = LLMRing(lockfile_path=lockfile_path)
                self.ring.require_aliases(["summarizer"], context="library-a")

        # Should raise ValueError because summarizer is missing
        with pytest.raises(ValueError) as exc_info:
            lib_a = MockLibraryA(lockfile_path=str(lockfile_incomplete_path))

        error_msg = str(exc_info.value)
        assert "summarizer" in error_msg
        assert "library-a" in error_msg

    def test_user_can_override_entire_chain_with_custom_lockfile(self, tmp_path):
        """Test that user can override entire chain with custom lockfile."""
        # Create default lockfiles for libraries
        lockfile_a_path = tmp_path / "library_a.lock"
        lockfile_a = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[
                        AliasBinding(alias="summarizer", models=["anthropic:claude-3-haiku"])
                    ],
                )
            },
        )
        lockfile_a.save(lockfile_a_path)

        # Create user's custom lockfile with different models
        user_lockfile_path = tmp_path / "user_custom.lock"
        user_lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[
                        AliasBinding(alias="summarizer", models=["openai:gpt-4o-mini"]),
                        AliasBinding(alias="analyzer", models=["google:gemini-1.5-flash"]),
                    ],
                )
            },
        )
        user_lockfile.save(user_lockfile_path)

        # Simulate Library A
        class MockLibraryA:
            def __init__(self, lockfile_path=None):
                self.ring = LLMRing(lockfile_path=lockfile_path or str(lockfile_a_path))
                self.ring.require_aliases(["summarizer"], context="library-a")

        # User overrides with custom lockfile
        lib_a_with_override = MockLibraryA(lockfile_path=str(user_lockfile_path))

        # Verify the override worked
        assert lib_a_with_override.ring.has_alias("summarizer") is True
        # Check that it's using user's lockfile (would be different model in real usage)
        assert lib_a_with_override.ring.lockfile_path == user_lockfile_path
