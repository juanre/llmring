"""Unit tests for ValidationService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmring.registry import RegistryModel
from llmring.schemas import LLMRequest, Message
from llmring.services.validation_service import ValidationService


class TestValidationService:
    """Tests for ValidationService."""

    @pytest.fixture
    def registry(self):
        """Create a mock registry."""
        return MagicMock()

    @pytest.fixture
    def validator(self, registry):
        """Create a ValidationService instance."""
        return ValidationService(registry)

    @pytest.fixture
    def sample_registry_model(self):
        """Sample registry model with limits."""
        return RegistryModel(
            provider="openai",
            model_name="gpt-4",
            display_name="GPT-4",
            description="Test model",
            max_input_tokens=8192,
            max_output_tokens=4096,
            supports_vision=True,
            supports_function_calling=True,
            supports_json_mode=True,
            is_active=True,
        )

    @pytest.mark.asyncio
    async def test_validate_context_limit_within_limits(self, validator, sample_registry_model):
        """Should pass validation when within limits."""
        request = LLMRequest(
            messages=[Message(role="user", content="Hello" * 100)],  # 500 chars
            model="openai:gpt-4",
        )

        with patch.object(validator, "_estimate_input_tokens", return_value=500):
            error = await validator.validate_context_limit(request, sample_registry_model)

        assert error is None

    @pytest.mark.asyncio
    async def test_validate_context_limit_exceeds_input(self, validator, sample_registry_model):
        """Should fail when input tokens exceed limit."""
        request = LLMRequest(
            messages=[Message(role="user", content="Hello" * 10000)],
            model="openai:gpt-4",
        )

        with patch.object(validator, "_estimate_input_tokens", return_value=10000):
            error = await validator.validate_context_limit(request, sample_registry_model)

        assert error is not None
        assert "Estimated input tokens (10000) exceeds" in error
        assert "8192" in error

    @pytest.mark.asyncio
    async def test_validate_context_limit_exceeds_output(self, validator, sample_registry_model):
        """Should fail when requested max_tokens exceeds output limit."""
        request = LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="openai:gpt-4",
            max_tokens=5000,  # Exceeds 4096 limit
        )

        with patch.object(validator, "_estimate_input_tokens", return_value=100):
            error = await validator.validate_context_limit(request, sample_registry_model)

        assert error is not None
        assert "Requested max tokens (5000) exceeds" in error
        assert "4096" in error

    @pytest.mark.asyncio
    async def test_validate_context_limit_no_model(self, validator):
        """Should skip validation when no model specified."""
        request = LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="",  # empty model -> treated as no model by validator
        )

        error = await validator.validate_context_limit(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_context_limit_invalid_model_format(self, validator):
        """Should skip validation for invalid model format."""
        request = LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="gpt-4",  # Missing provider prefix
        )

        error = await validator.validate_context_limit(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_context_limit_no_registry_model(self, validator):
        """Should skip validation when registry model not found."""
        request = LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="openai:gpt-4",
        )

        validator.registry.fetch_current_models = AsyncMock(return_value=[])

        error = await validator.validate_context_limit(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_context_limit_no_limits_in_registry(self, validator):
        """Should skip validation when model has no limits."""
        registry_model = RegistryModel(
            provider="openai",
            model_name="gpt-4",
            display_name="GPT-4",
            description="Test model",
            max_input_tokens=None,  # No limit
            max_output_tokens=None,
            is_active=True,
        )

        request = LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="openai:gpt-4",
        )

        error = await validator.validate_context_limit(request, registry_model)
        assert error is None

    def test_estimate_input_tokens_large_input(self, validator, sample_registry_model):
        """Should use character count for very large inputs."""
        request = LLMRequest(
            messages=[Message(role="user", content="X" * 20000)],
            model="openai:gpt-4",
        )

        # Should skip tokenization and return char count
        tokens = validator._estimate_input_tokens(request, "openai", "gpt-4", sample_registry_model)

        assert tokens == 20000  # Character count

    def test_estimate_input_tokens_normal_input(self, validator, sample_registry_model):
        """Should use tokenization for normal-sized inputs."""
        request = LLMRequest(
            messages=[Message(role="user", content="Hello, world!")],
            model="openai:gpt-4",
        )

        with patch("llmring.token_counter.count_tokens", return_value=4):
            tokens = validator._estimate_input_tokens(
                request, "openai", "gpt-4", sample_registry_model
            )

        assert tokens == 4

    def test_estimate_input_tokens_tokenization_fails(self, validator, sample_registry_model):
        """Should fall back to character count if tokenization fails."""
        request = LLMRequest(
            messages=[Message(role="user", content="Hello, world!")],
            model="openai:gpt-4",
        )

        with patch(
            "llmring.token_counter.count_tokens",
            side_effect=Exception("Tokenization failed"),
        ):
            tokens = validator._estimate_input_tokens(
                request, "openai", "gpt-4", sample_registry_model
            )

        assert tokens == 13  # Character count of "Hello, world!"

    def test_estimate_input_tokens_list_content(self, validator, sample_registry_model):
        """Should handle list content in messages."""
        request = LLMRequest(
            messages=[
                Message(
                    role="user",
                    content=[{"type": "text", "text": "Hello"}],
                )
            ],
            model="openai:gpt-4",
        )

        with patch("llmring.token_counter.count_tokens", return_value=2):
            tokens = validator._estimate_input_tokens(
                request, "openai", "gpt-4", sample_registry_model
            )

        assert tokens == 2

    @pytest.mark.asyncio
    async def test_validate_model_capabilities_vision_supported(
        self, validator, sample_registry_model
    ):
        """Should pass when model supports vision."""
        request = LLMRequest(
            messages=[
                Message(
                    role="user",
                    content=[
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
                    ],
                )
            ],
            model="openai:gpt-4",
        )

        error = await validator.validate_model_capabilities(request, sample_registry_model)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_model_capabilities_vision_not_supported(self, validator):
        """Should fail when model doesn't support vision."""
        registry_model = RegistryModel(
            provider="openai",
            model_name="gpt-3.5-turbo",
            display_name="GPT-3.5 Turbo",
            description="Test model",
            max_input_tokens=4096,
            supports_vision=False,  # No vision support
            is_active=True,
        )

        request = LLMRequest(
            messages=[
                Message(
                    role="user",
                    content=[
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image", "data": "base64data"},
                    ],
                )
            ],
            model="openai:gpt-3.5-turbo",
        )

        error = await validator.validate_model_capabilities(request, registry_model)
        assert error is not None
        assert "does not support vision" in error

    @pytest.mark.asyncio
    async def test_validate_model_capabilities_function_calling_supported(
        self, validator, sample_registry_model
    ):
        """Should pass when model supports function calling."""
        request = LLMRequest(
            messages=[Message(role="user", content="Call a function")],
            model="openai:gpt-4",
            tools=[{"type": "function", "function": {"name": "test_func"}}],
        )

        error = await validator.validate_model_capabilities(request, sample_registry_model)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_model_capabilities_function_calling_not_supported(self, validator):
        """Should fail when model doesn't support function calling."""
        registry_model = RegistryModel(
            provider="openai",
            model_name="gpt-3",
            display_name="GPT-3",
            description="Test model",
            max_input_tokens=2048,
            supports_function_calling=False,  # No function calling
            is_active=True,
        )

        request = LLMRequest(
            messages=[Message(role="user", content="Call a function")],
            model="openai:gpt-3",
            tools=[{"type": "function", "function": {"name": "test_func"}}],
        )

        error = await validator.validate_model_capabilities(request, registry_model)
        assert error is not None
        assert "does not support function calling" in error

    @pytest.mark.asyncio
    async def test_validate_model_capabilities_json_mode_supported(
        self, validator, sample_registry_model
    ):
        """Should pass when model supports JSON mode."""
        request = LLMRequest(
            messages=[Message(role="user", content="Return JSON")],
            model="openai:gpt-4",
            json_response=True,
        )

        error = await validator.validate_model_capabilities(request, sample_registry_model)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_model_capabilities_json_mode_not_supported(self, validator):
        """Should fail when model doesn't support JSON mode."""
        registry_model = RegistryModel(
            provider="openai",
            model_name="gpt-3",
            display_name="GPT-3",
            description="Test model",
            max_input_tokens=2048,
            supports_json_mode=False,  # No JSON mode
            is_active=True,
        )

        request = LLMRequest(
            messages=[Message(role="user", content="Return JSON")],
            model="openai:gpt-3",
            json_response=True,
        )

        error = await validator.validate_model_capabilities(request, registry_model)
        assert error is not None
        assert "does not support JSON mode" in error

    @pytest.mark.asyncio
    async def test_validate_model_capabilities_no_model(self, validator):
        """Should skip validation when no model specified."""
        request = LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="",  # empty model -> treated as no model by validator
        )

        error = await validator.validate_model_capabilities(request)
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_model_capabilities_no_registry_model(self, validator):
        """Should skip validation when registry model not found."""
        request = LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="openai:gpt-4",
        )

        validator.registry.fetch_current_models = AsyncMock(return_value=[])

        error = await validator.validate_model_capabilities(request)
        assert error is None
