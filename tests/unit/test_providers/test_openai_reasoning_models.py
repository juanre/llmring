"""
ABOUTME: Unit tests for OpenAI reasoning model support.
ABOUTME: Tests registry-based detection and token handling for reasoning models.
"""

import pytest

from llmring.providers.openai_api import OpenAIProvider
from llmring.schemas import Message


@pytest.mark.unit
class TestOpenAIReasoningModels:
    """Test reasoning model detection and token handling."""

    @pytest.mark.asyncio
    async def test_get_model_config_reasoning_model(self, openai_provider):
        """Test that gpt-5 is detected as a reasoning model from registry."""
        config = await openai_provider._get_model_config("gpt-5")

        assert config["is_reasoning"] is True
        assert config["min_recommended_reasoning_tokens"] == 2000
        assert config["api_endpoint"] == "chat"

    @pytest.mark.asyncio
    async def test_get_model_config_non_reasoning_model(self, openai_provider):
        """Test that gpt-4 is not detected as a reasoning model."""
        config = await openai_provider._get_model_config("gpt-4")

        assert config["is_reasoning"] is False
        assert config["min_recommended_reasoning_tokens"] is None

    @pytest.mark.asyncio
    async def test_get_model_config_o1_model(self, openai_provider):
        """Test that o1 is detected as a reasoning model."""
        config = await openai_provider._get_model_config("o1")

        assert config["is_reasoning"] is True
        assert config["min_recommended_reasoning_tokens"] == 2000

    @pytest.mark.asyncio
    async def test_get_model_config_fallback_when_registry_unavailable(self):
        """Test fallback to string matching when registry is unavailable."""
        provider = OpenAIProvider(api_key="test-key")
        provider._registry_client = None

        # Should fall back to string matching
        config = await provider._get_model_config("gpt-5")
        assert config["is_reasoning"] is True
        assert config["min_recommended_reasoning_tokens"] == 2000

        config = await provider._get_model_config("o1")
        assert config["is_reasoning"] is True

        config = await provider._get_model_config("gpt-4")
        assert config["is_reasoning"] is False

    @pytest.mark.asyncio
    async def test_get_model_config_unknown_model_fallback(self, openai_provider):
        """Test that unknown models fall back to non-reasoning."""
        config = await openai_provider._get_model_config("unknown-model-xyz")

        # Unknown models should default to non-reasoning
        assert config["is_reasoning"] is False
        assert config["min_recommended_reasoning_tokens"] is None

    @pytest.mark.asyncio
    @pytest.mark.llm
    async def test_reasoning_model_automatic_token_handling(self, openai_provider):
        """Test automatic reasoning token budget for reasoning models."""
        messages = [Message(role="user", content="Say 'hello'")]

        # For reasoning model with low max_tokens, should add reasoning budget
        response = await openai_provider.chat(
            messages=messages,
            model="gpt-5",
            max_tokens=500  # User wants 500 output tokens
            # Should automatically add 2000 reasoning tokens = 2500 total
        )

        # Response should not be empty
        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    @pytest.mark.llm
    async def test_reasoning_model_explicit_reasoning_tokens(self, openai_provider):
        """Test explicit reasoning_tokens parameter."""
        messages = [Message(role="user", content="Say 'hello'")]

        # User explicitly sets reasoning budget
        response = await openai_provider.chat(
            messages=messages,
            model="gpt-5",
            max_tokens=500,
            reasoning_tokens=3000  # Custom reasoning budget
            # Should use 3000 + 500 = 3500 total
        )

        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    @pytest.mark.llm
    async def test_non_reasoning_model_ignores_reasoning_tokens(self, openai_provider):
        """Test that non-reasoning models ignore reasoning_tokens parameter."""
        messages = [Message(role="user", content="Say 'hello'")]

        # reasoning_tokens should be ignored for non-reasoning models
        response = await openai_provider.chat(
            messages=messages,
            model="gpt-3.5-turbo",
            max_tokens=50,
            reasoning_tokens=2000  # Should be ignored
        )

        assert response.content is not None
        assert len(response.content) > 0
        # Usage should reflect only max_tokens, not reasoning_tokens
        assert response.usage is not None
