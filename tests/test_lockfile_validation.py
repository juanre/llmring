"""Tests for lockfile validation helpers (has_alias, require_aliases).

These tests verify the library pattern where libraries ship with lockfiles
and validate required aliases on initialization.
"""

from pathlib import Path

import pytest

from llmring.lockfile_core import AliasBinding, Lockfile, ProfileConfig
from llmring.service import LLMRing


class TestLockfileHasAlias:
    """Tests for Lockfile.has_alias() method."""

    def test_returns_true_when_alias_exists_in_default_profile(self):
        """has_alias returns True for existing alias in default profile."""
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

        assert lockfile.has_alias("summarizer") is True

    def test_returns_true_when_alias_exists_in_specified_profile(self):
        """has_alias returns True for existing alias in specified profile."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "dev": ProfileConfig(
                    name="dev",
                    bindings=[AliasBinding(alias="cheap_model", models=["openai:gpt-4o-mini"])],
                )
            },
        )

        assert lockfile.has_alias("cheap_model", profile="dev") is True

    def test_returns_false_when_alias_doesnt_exist(self):
        """has_alias returns False for non-existent alias."""
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

    def test_returns_false_when_profile_doesnt_exist(self):
        """has_alias returns False when querying non-existent profile."""
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

        assert lockfile.has_alias("summarizer", profile="nonexistent") is False

    def test_returns_false_when_lockfile_has_no_bindings(self):
        """has_alias returns False when profile has no bindings."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={"default": ProfileConfig(name="default", bindings=[])},
        )

        assert lockfile.has_alias("summarizer") is False

    def test_handles_none_profile_uses_default(self):
        """has_alias with None profile uses default_profile."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="prod",
            profiles={
                "prod": ProfileConfig(
                    name="prod",
                    bindings=[
                        AliasBinding(alias="summarizer", models=["anthropic:claude-3-5-sonnet"])
                    ],
                )
            },
        )

        # None should use default_profile which is "prod"
        assert lockfile.has_alias("summarizer", profile=None) is True


class TestLockfileRequireAliases:
    """Tests for Lockfile.require_aliases() method."""

    def test_passes_when_all_required_aliases_exist(self):
        """require_aliases passes when all aliases exist."""
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

        # Should not raise
        lockfile.require_aliases(["summarizer", "analyzer"])

    def test_raises_valueerror_when_one_alias_missing(self):
        """require_aliases raises ValueError listing the missing alias."""
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
            lockfile.require_aliases(["summarizer", "missing_alias"])

        error_msg = str(exc_info.value)
        assert "missing_alias" in error_msg
        assert "required aliases" in error_msg.lower()

    def test_raises_valueerror_when_multiple_aliases_missing(self):
        """require_aliases lists all missing aliases in error."""
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
            lockfile.require_aliases(["missing1", "missing2", "missing3"])

        error_msg = str(exc_info.value)
        assert "missing1" in error_msg
        assert "missing2" in error_msg
        assert "missing3" in error_msg

    def test_error_message_includes_lockfile_reference(self):
        """require_aliases error mentions lockfile for context."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={"default": ProfileConfig(name="default", bindings=[])},
        )
        lockfile.file_path = Path("/path/to/test.lock")

        with pytest.raises(ValueError) as exc_info:
            lockfile.require_aliases(["missing"])

        error_msg = str(exc_info.value)
        assert "lockfile" in error_msg.lower() or "/path/to/test.lock" in error_msg

    def test_error_message_includes_context_if_provided(self):
        """require_aliases includes context (library name) in error."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={"default": ProfileConfig(name="default", bindings=[])},
        )

        with pytest.raises(ValueError) as exc_info:
            lockfile.require_aliases(["missing"], context="test-library")

        error_msg = str(exc_info.value)
        assert "test-library" in error_msg

    def test_error_message_is_helpful_and_actionable(self):
        """require_aliases provides helpful guidance."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={"default": ProfileConfig(name="default", bindings=[])},
        )

        with pytest.raises(ValueError) as exc_info:
            lockfile.require_aliases(["summarizer"])

        error_msg = str(exc_info.value)
        # Should mention what's missing
        assert "summarizer" in error_msg
        # Should be actionable (suggest what to do)
        assert "lockfile" in error_msg.lower() or "define" in error_msg.lower()

    def test_works_with_empty_required_list(self):
        """require_aliases with empty list is a no-op (doesn't raise)."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={"default": ProfileConfig(name="default", bindings=[])},
        )

        # Should not raise
        lockfile.require_aliases([])

    def test_works_with_specified_profile(self):
        """require_aliases validates against specified profile."""
        lockfile = Lockfile(
            version="1.0",
            default_profile="default",
            profiles={
                "default": ProfileConfig(
                    name="default",
                    bindings=[AliasBinding(alias="default_alias", models=["openai:gpt-4o"])],
                ),
                "prod": ProfileConfig(
                    name="prod",
                    bindings=[
                        AliasBinding(alias="prod_alias", models=["anthropic:claude-3-5-sonnet"])
                    ],
                ),
            },
        )

        # Should pass for prod profile
        lockfile.require_aliases(["prod_alias"], profile="prod")

        # Should fail for default profile (prod_alias not there)
        with pytest.raises(ValueError):
            lockfile.require_aliases(["prod_alias"], profile="default")


class TestLLMRingValidationProxies:
    """Tests for LLMRing validation proxy methods."""

    def test_has_alias_proxies_to_lockfile(self):
        """LLMRing.has_alias() proxies correctly to lockfile."""
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

        ring = LLMRing.__new__(LLMRing)  # Create without __init__
        ring.lockfile = lockfile

        assert ring.has_alias("summarizer") is True
        assert ring.has_alias("nonexistent") is False

    def test_require_aliases_proxies_to_lockfile(self):
        """LLMRing.require_aliases() proxies correctly to lockfile."""
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

        ring = LLMRing.__new__(LLMRing)
        ring.lockfile = lockfile

        # Should pass
        ring.require_aliases(["summarizer"])

        # Should raise
        with pytest.raises(ValueError):
            ring.require_aliases(["missing"])

    def test_has_alias_returns_false_if_no_lockfile_loaded(self):
        """LLMRing.has_alias() returns False if no lockfile."""
        ring = LLMRing.__new__(LLMRing)
        ring.lockfile = None

        assert ring.has_alias("anything") is False

    def test_require_aliases_raises_clear_error_if_no_lockfile(self):
        """LLMRing.require_aliases() raises clear error if no lockfile."""
        ring = LLMRing.__new__(LLMRing)
        ring.lockfile = None

        with pytest.raises(ValueError) as exc_info:
            ring.require_aliases(["summarizer"])

        error_msg = str(exc_info.value)
        assert "no lockfile" in error_msg.lower()
        assert "summarizer" in error_msg


class TestLibraryCompositionPattern:
    """Integration tests for library composition pattern."""

    def test_library_with_bundled_lockfile(self, tmp_path):
        """Library can use its bundled lockfile."""
        # Create a library lockfile
        lib_lockfile = tmp_path / "library.lock"
        lib_lockfile.write_text(
            """
version = "1.0"
default_profile = "default"

[profiles.default]
name = "default"

[[profiles.default.bindings]]
alias = "summarizer"
models = ["openai:gpt-4o-mini"]
"""
        )

        # Library initializes with its lockfile
        ring = LLMRing(lockfile_path=str(lib_lockfile))

        # Validate required aliases
        ring.require_aliases(["summarizer"], context="test-library")

        # Should work
        models = ring.lockfile.resolve_alias("summarizer")
        assert models == ["openai:gpt-4o-mini"]

    def test_user_can_override_library_lockfile(self, tmp_path):
        """User can override library's lockfile with custom one."""
        # Library's lockfile
        lib_lockfile = tmp_path / "library.lock"
        lib_lockfile.write_text(
            """
version = "1.0"
default_profile = "default"

[profiles.default]
name = "default"

[[profiles.default.bindings]]
alias = "summarizer"
models = ["openai:gpt-4o-mini"]
"""
        )

        # User's custom lockfile with different model
        user_lockfile = tmp_path / "user.lock"
        user_lockfile.write_text(
            """
version = "1.0"
default_profile = "default"

[profiles.default]
name = "default"

[[profiles.default.bindings]]
alias = "summarizer"
models = ["anthropic:claude-3-5-sonnet"]
"""
        )

        # User initializes library with their lockfile
        ring = LLMRing(lockfile_path=str(user_lockfile))
        ring.require_aliases(["summarizer"], context="test-library")

        # Should use user's model choice
        models = ring.lockfile.resolve_alias("summarizer")
        assert models == ["anthropic:claude-3-5-sonnet"]

    def test_library_composition_shared_lockfile(self, tmp_path):
        """Library B using Library A can share a lockfile."""
        # Shared lockfile with aliases for both libraries
        shared_lockfile = tmp_path / "shared.lock"
        shared_lockfile.write_text(
            """
version = "1.0"
default_profile = "default"

[profiles.default]
name = "default"

[[profiles.default.bindings]]
alias = "summarizer"
models = ["openai:gpt-4o"]

[[profiles.default.bindings]]
alias = "analyzer"
models = ["anthropic:claude-3-5-sonnet"]
"""
        )

        # Simulate Library A
        ring_a = LLMRing(lockfile_path=str(shared_lockfile))
        ring_a.require_aliases(["summarizer"], context="library-a")

        # Simulate Library B using same lockfile
        ring_b = LLMRing(lockfile_path=str(shared_lockfile))
        ring_b.require_aliases(["analyzer"], context="library-b")

        # Both should work
        assert ring_a.lockfile.resolve_alias("summarizer") == ["openai:gpt-4o"]
        assert ring_b.lockfile.resolve_alias("analyzer") == ["anthropic:claude-3-5-sonnet"]

    def test_validation_fails_with_helpful_error_for_missing_library_alias(self, tmp_path):
        """When user's lockfile is incomplete, error is clear."""
        # User's lockfile missing Library A's required alias
        incomplete_lockfile = tmp_path / "incomplete.lock"
        incomplete_lockfile.write_text(
            """
version = "1.0"
default_profile = "default"

[profiles.default]
name = "default"

[[profiles.default.bindings]]
alias = "my_alias"
models = ["openai:gpt-4o"]
"""
        )

        # Library tries to initialize
        ring = LLMRing(lockfile_path=str(incomplete_lockfile))

        with pytest.raises(ValueError) as exc_info:
            ring.require_aliases(["summarizer", "analyzer"], context="library-a")

        error_msg = str(exc_info.value)
        # Should mention both missing aliases
        assert "summarizer" in error_msg
        assert "analyzer" in error_msg
        # Should mention the library context
        assert "library-a" in error_msg
        # Should be actionable
        assert "lockfile" in error_msg.lower()
