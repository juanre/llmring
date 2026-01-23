"""Unit tests for model parsing utilities."""

import pytest

from llmring.utils import is_model_reference, parse_model_string


class TestIsModelReference:
    """Tests for is_model_reference function."""

    def test_known_providers_return_true(self):
        """Should return True for known provider prefixes."""
        assert is_model_reference("openai:gpt-4") is True
        assert is_model_reference("anthropic:claude-3-opus") is True
        assert is_model_reference("google:gemini-pro") is True
        assert is_model_reference("ollama:llama2") is True

    def test_unknown_namespace_returns_false(self):
        """Should return False for unknown namespaces (library aliases)."""
        assert is_model_reference("libA:summarizer") is False
        assert is_model_reference("my-package:fast") is False

    def test_no_colon_returns_false(self):
        """Should return False for strings without colon."""
        assert is_model_reference("fast") is False
        assert is_model_reference("summarizer") is False

    def test_case_insensitive(self):
        """Should be case-insensitive for provider prefix."""
        assert is_model_reference("OpenAI:gpt-4") is True
        assert is_model_reference("ANTHROPIC:claude") is True
        assert is_model_reference("Google:gemini") is True

    def test_empty_string_returns_false(self):
        """Should return False for empty string."""
        assert is_model_reference("") is False

    def test_colon_only_returns_false(self):
        """Should return False for string with only colon."""
        assert is_model_reference(":") is False
        assert is_model_reference(":model") is False

    def test_multiple_colons_checks_first_segment(self):
        """Should only check first segment when multiple colons present."""
        assert is_model_reference("openai:gpt-4:extra") is True
        assert is_model_reference("libA:alias:extra") is False


class TestParseModelString:
    """Tests for parse_model_string function."""

    def test_valid_format(self):
        """Should parse valid provider:model format."""
        provider, model = parse_model_string("openai:gpt-4")
        assert provider == "openai"
        assert model == "gpt-4"

    def test_preserves_case(self):
        """Should preserve case in provider and model names."""
        provider, model = parse_model_string("OpenAI:GPT-4")
        assert provider == "OpenAI"
        assert model == "GPT-4"

    def test_multiple_colons(self):
        """Should handle model names with colons."""
        provider, model = parse_model_string("anthropic:claude-3:beta")
        assert provider == "anthropic"
        assert model == "claude-3:beta"

    def test_no_colon_raises(self):
        """Should raise ValueError for missing colon."""
        with pytest.raises(ValueError, match="provider:model"):
            parse_model_string("gpt-4")

    def test_empty_provider_raises(self):
        """Should raise ValueError for empty provider."""
        with pytest.raises(ValueError, match="non-empty"):
            parse_model_string(":gpt-4")

    def test_empty_model_raises(self):
        """Should raise ValueError for empty model."""
        with pytest.raises(ValueError, match="non-empty"):
            parse_model_string("openai:")
