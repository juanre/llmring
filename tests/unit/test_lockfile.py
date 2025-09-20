"""
Unit tests for lockfile functionality.

Tests lockfile creation, loading, alias management, and profile support.
"""

import json
import os
from unittest.mock import patch

import pytest

from llmring.lockfile import AliasBinding, Lockfile, ProfileConfig


class TestLockfile:
    """Test Lockfile class functionality."""

    def test_create_default_no_api_keys(self):
        """Test creating default lockfile with no API keys."""
        with patch.dict(os.environ, {}, clear=True):
            lockfile = Lockfile.create_default()

            assert lockfile.version == "1.0"
            assert lockfile.default_profile == "default"
            assert "default" in lockfile.profiles
            assert "prod" in lockfile.profiles
            assert "staging" in lockfile.profiles
            assert "dev" in lockfile.profiles

            # Should have minimal bindings without API keys (only local ollama)
            default_profile = lockfile.get_profile("default")
            assert (
                len(default_profile.bindings) == 1
            )  # Only local ollama alias without hardcoded models
            # Verify the local alias uses the correct model
            local_binding = next(
                (b for b in default_profile.bindings if b.alias == "local"), None
            )
            assert local_binding is not None
            assert local_binding.model == "llama3:latest"

    def test_create_default_with_openai_key(self):
        """Test creating default lockfile with OpenAI API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            lockfile = Lockfile.create_default()

            default_profile = lockfile.get_profile("default")
            aliases = {b.alias for b in default_profile.bindings}

            # Without registry data, basic create_default() doesn't add provider-specific bindings
            # Only local ollama binding is added
            assert "local" in aliases
            assert len(default_profile.bindings) == 1

    def test_create_default_with_anthropic_key(self):
        """Test creating default lockfile with Anthropic API key."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
            lockfile = Lockfile.create_default()

            default_profile = lockfile.get_profile("default")
            aliases = {b.alias for b in default_profile.bindings}

            # Without registry data, basic create_default() doesn't add provider-specific bindings
            # Only local ollama binding is added
            assert "local" in aliases
            assert len(default_profile.bindings) == 1

            # Check correct model assignments
            for binding in default_profile.bindings:
                if binding.alias == "deep":
                    assert binding.provider == "anthropic"
                    assert "opus" in binding.model or "sonnet" in binding.model

    def test_save_and_load_toml(self, tmp_path):
        """Test saving and loading lockfile in TOML format."""
        lockfile = Lockfile.create_default()
        lockfile.set_binding("test_alias", "openai:gpt-4", profile="default")

        # Save as TOML
        lockfile_path = tmp_path / "llmring.lock"
        lockfile.save(lockfile_path)

        assert lockfile_path.exists()

        # Load and verify
        loaded = Lockfile.load(lockfile_path)
        assert loaded.version == lockfile.version
        assert loaded.default_profile == lockfile.default_profile

        # Check binding was preserved
        binding = loaded.get_binding("test_alias", profile="default")
        assert binding is not None
        assert binding.model_ref == "openai:gpt-4"

    def test_save_and_load_json(self, tmp_path):
        """Test saving and loading lockfile in JSON format."""
        lockfile = Lockfile.create_default()
        lockfile.set_binding("test_alias", "anthropic:claude-3-opus", profile="default")

        # Save as JSON
        lockfile_path = tmp_path / "llmring.lock.json"
        lockfile.save(lockfile_path)

        assert lockfile_path.exists()

        # Verify it's valid JSON
        with open(lockfile_path) as f:
            data = json.load(f)
            assert data["version"] == "1.0"

        # Load and verify
        loaded = Lockfile.load(lockfile_path)
        binding = loaded.get_binding("test_alias", profile="default")
        assert binding.model_ref == "anthropic:claude-3-opus"

    def test_set_and_get_binding(self):
        """Test setting and getting bindings."""
        lockfile = Lockfile()

        # Set binding
        lockfile.set_binding("summarizer", "openai:gpt-3.5-turbo", profile="default")

        # Get binding
        binding = lockfile.get_binding("summarizer", profile="default")
        assert binding is not None
        assert binding.alias == "summarizer"
        assert binding.provider == "openai"
        assert binding.model == "gpt-3.5-turbo"
        assert binding.model_ref == "openai:gpt-3.5-turbo"

    def test_update_existing_binding(self):
        """Test updating an existing binding."""
        lockfile = Lockfile()

        # Set initial binding
        lockfile.set_binding("summarizer", "openai:gpt-3.5-turbo")

        # Update binding
        lockfile.set_binding("summarizer", "openai:gpt-4")

        # Should have only one binding with updated value
        profile = lockfile.get_profile("default")
        assert len(profile.bindings) == 1
        assert profile.bindings[0].model == "gpt-4"

    def test_remove_binding(self):
        """Test removing a binding."""
        lockfile = Lockfile()
        lockfile.set_binding("test1", "openai:gpt-3.5-turbo")
        lockfile.set_binding("test2", "anthropic:claude-3-haiku")

        profile = lockfile.get_profile("default")
        assert len(profile.bindings) == 2

        # Remove one binding
        removed = profile.remove_binding("test1")
        assert removed is True
        assert len(profile.bindings) == 1
        assert profile.bindings[0].alias == "test2"

        # Try to remove non-existent
        removed = profile.remove_binding("test1")
        assert removed is False

    def test_list_aliases(self):
        """Test listing aliases."""
        lockfile = Lockfile()
        lockfile.set_binding("alias1", "openai:gpt-3.5-turbo")
        lockfile.set_binding("alias2", "anthropic:claude-3-haiku")
        lockfile.set_binding("alias3", "google:gemini-pro")

        aliases = lockfile.list_aliases(profile="default")
        assert aliases == ["alias1", "alias2", "alias3"]

    def test_resolve_alias(self):
        """Test resolving aliases."""
        lockfile = Lockfile()
        lockfile.set_binding("summarizer", "openai:gpt-3.5-turbo")

        # Resolve existing alias
        model_ref = lockfile.resolve_alias("summarizer")
        assert model_ref == "openai:gpt-3.5-turbo"

        # Try to resolve non-existent alias
        model_ref = lockfile.resolve_alias("non_existent")
        assert model_ref is None

    def test_profile_management(self):
        """Test profile creation and management."""
        lockfile = Lockfile()

        # Set bindings in different profiles
        lockfile.set_binding("test", "openai:gpt-3.5-turbo", profile="dev")
        lockfile.set_binding("test", "openai:gpt-4", profile="prod")

        # Check each profile has correct binding
        dev_binding = lockfile.get_binding("test", profile="dev")
        assert dev_binding.model == "gpt-3.5-turbo"

        prod_binding = lockfile.get_binding("test", profile="prod")
        assert prod_binding.model == "gpt-4"

    def test_get_profile_creates_if_missing(self):
        """Test that get_profile creates profile if it doesn't exist."""
        lockfile = Lockfile()

        # Profile shouldn't exist yet
        assert "custom" not in lockfile.profiles

        # Getting it should create it
        profile = lockfile.get_profile("custom")
        assert profile is not None
        assert profile.name == "custom"
        assert "custom" in lockfile.profiles

    def test_find_lockfile(self, tmp_path):
        """Test finding lockfile in parent directories."""
        # Create directory structure
        root = tmp_path / "project"
        subdir = root / "src" / "component"
        subdir.mkdir(parents=True)

        # Create lockfile in root
        lockfile_path = root / "llmring.lock"
        lockfile = Lockfile.create_default()
        lockfile.save(lockfile_path)

        # Should find lockfile from subdirectory
        with patch("os.getcwd", return_value=str(subdir)):
            found_path = Lockfile.find_lockfile()
            assert found_path is not None
            assert found_path.name == "llmring.lock"
            assert found_path.parent == root

    def test_find_lockfile_not_found(self, tmp_path):
        """Test when lockfile is not found."""
        with patch("os.getcwd", return_value=str(tmp_path)):
            found_path = Lockfile.find_lockfile()
            assert found_path is None

    def test_constraints_in_binding(self):
        """Test bindings with constraints."""
        lockfile = Lockfile()
        constraints = {"max_tokens": 1000, "temperature": 0.5}

        lockfile.set_binding(
            "constrained",
            "openai:gpt-3.5-turbo",
            profile="default",
            constraints=constraints,
        )

        binding = lockfile.get_binding("constrained")
        assert binding.constraints == constraints

    def test_model_ref_parsing(self):
        """Test parsing model references."""
        # Valid model ref
        binding = AliasBinding.from_model_ref("test", "openai:gpt-4")
        assert binding.provider == "openai"
        assert binding.model == "gpt-4"
        assert binding.model_ref == "openai:gpt-4"

        # Invalid model ref (no colon)
        with pytest.raises(ValueError, match="Invalid model reference"):
            AliasBinding.from_model_ref("test", "gpt-4")

    def test_timestamps(self, tmp_path):
        """Test that timestamps are properly managed."""
        import time

        lockfile = Lockfile()
        created_at = lockfile.created_at

        # Wait a moment and update
        time.sleep(0.01)
        lockfile.set_binding("test", "openai:gpt-4")

        # Save and reload
        lockfile_path = tmp_path / "llmring.lock"
        lockfile.save(lockfile_path)

        loaded = Lockfile.load(lockfile_path)

        # Created time should be preserved
        assert loaded.created_at == created_at
        # Updated time should be recent
        assert loaded.updated_at > created_at


class TestProfileConfig:
    """Test ProfileConfig class functionality."""

    def test_profile_creation(self):
        """Test creating a profile."""
        profile = ProfileConfig(name="test_profile")
        assert profile.name == "test_profile"
        assert profile.bindings == []
        assert profile.registry_versions == {}

    def test_get_binding(self):
        """Test getting a specific binding from profile."""
        profile = ProfileConfig(name="test")
        binding = AliasBinding(alias="test", provider="openai", model="gpt-4")
        profile.bindings.append(binding)

        found = profile.get_binding("test")
        assert found == binding

        not_found = profile.get_binding("non_existent")
        assert not_found is None

    def test_set_binding(self):
        """Test setting bindings in profile."""
        profile = ProfileConfig(name="test")

        # Set new binding
        profile.set_binding("alias1", "openai:gpt-3.5-turbo")
        assert len(profile.bindings) == 1
        assert profile.bindings[0].alias == "alias1"

        # Update existing binding
        profile.set_binding("alias1", "openai:gpt-4")
        assert len(profile.bindings) == 1  # Should replace, not add
        assert profile.bindings[0].model == "gpt-4"

        # Add another binding
        profile.set_binding("alias2", "anthropic:claude-3-haiku")
        assert len(profile.bindings) == 2


class TestAliasBinding:
    """Test AliasBinding class functionality."""

    def test_alias_binding_creation(self):
        """Test creating an alias binding."""
        binding = AliasBinding(
            alias="summarizer", provider="openai", model="gpt-3.5-turbo"
        )

        assert binding.alias == "summarizer"
        assert binding.provider == "openai"
        assert binding.model == "gpt-3.5-turbo"
        assert binding.model_ref == "openai:gpt-3.5-turbo"
        assert binding.constraints is None

    def test_alias_binding_with_constraints(self):
        """Test alias binding with constraints."""
        constraints = {"max_tokens": 500, "temperature": 0.7}
        binding = AliasBinding(
            alias="creative", provider="openai", model="gpt-4", constraints=constraints
        )

        assert binding.constraints == constraints

    def test_from_model_ref(self):
        """Test creating binding from model reference."""
        binding = AliasBinding.from_model_ref(
            "test_alias", "anthropic:claude-3-opus-20240229"
        )

        assert binding.alias == "test_alias"
        assert binding.provider == "anthropic"
        assert binding.model == "claude-3-opus-20240229"
        assert binding.model_ref == "anthropic:claude-3-opus-20240229"

    def test_from_model_ref_with_constraints(self):
        """Test creating binding from model ref with constraints."""
        constraints = {"temperature": 0.5}
        binding = AliasBinding.from_model_ref(
            "precise", "openai:gpt-4", constraints=constraints
        )

        assert binding.constraints == constraints

    def test_from_model_ref_invalid(self):
        """Test invalid model reference."""
        with pytest.raises(ValueError, match="Invalid model reference"):
            AliasBinding.from_model_ref("alias", "invalid_model_ref")

        with pytest.raises(ValueError, match="Invalid model reference"):
            AliasBinding.from_model_ref("alias", "")

        # Edge case: multiple colons
        binding = AliasBinding.from_model_ref("alias", "provider:model:version")
        assert binding.provider == "provider"
        assert binding.model == "model:version"  # Everything after first colon
