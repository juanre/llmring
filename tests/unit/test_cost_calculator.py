"""Unit tests for CostCalculator service."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llmring.registry import RegistryModel
from llmring.schemas import LLMResponse
from llmring.services.cost_calculator import CostCalculator


class TestCostCalculator:
    """Tests for CostCalculator service."""

    @pytest.fixture
    def registry(self):
        """Create a mock registry."""
        return MagicMock()

    @pytest.fixture
    def calculator(self, registry):
        """Create a CostCalculator instance."""
        return CostCalculator(registry)

    @pytest.fixture
    def sample_registry_model(self):
        """Sample registry model with pricing."""
        return RegistryModel(
            provider="openai",
            model_name="gpt-4",
            display_name="GPT-4",
            description="Test model",
            max_input_tokens=8192,
            max_output_tokens=4096,
            dollars_per_million_tokens_input=30.0,
            dollars_per_million_tokens_output=60.0,
            is_active=True,
        )

    @pytest.fixture
    def sample_response(self):
        """Sample LLM response with usage."""
        return LLMResponse(
            content="Hello, world!",
            model="openai:gpt-4",
            provider="openai",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        )

    @pytest.mark.asyncio
    async def test_calculate_cost_success(self, calculator, sample_response, sample_registry_model):
        """Should calculate cost correctly."""
        cost = await calculator.calculate_cost(sample_response, sample_registry_model)

        assert cost is not None
        # 100 tokens * $30/1M = $0.003
        assert cost["input_cost"] == pytest.approx(0.003)
        # 50 tokens * $60/1M = $0.003
        assert cost["output_cost"] == pytest.approx(0.003)
        # Total = $0.006
        assert cost["total_cost"] == pytest.approx(0.006)
        assert cost["cost_per_million_input"] == 30.0
        assert cost["cost_per_million_output"] == 60.0

    @pytest.mark.asyncio
    async def test_calculate_cost_no_usage(self, calculator, sample_registry_model):
        """Should return None when no usage info."""
        response = LLMResponse(
            content="Hello",
            model="openai:gpt-4",
            provider="openai",
        )

        cost = await calculator.calculate_cost(response, sample_registry_model)
        assert cost is None

    @pytest.mark.asyncio
    async def test_calculate_cost_invalid_model_format(self, calculator):
        """Should return None for invalid model format."""
        response = LLMResponse(
            content="Hello",
            model="gpt-4",  # Missing provider prefix
            provider="openai",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

        cost = await calculator.calculate_cost(response)
        assert cost is None

    @pytest.mark.asyncio
    async def test_calculate_cost_no_pricing(self, calculator):
        """Should return None when pricing not available."""
        registry_model = RegistryModel(
            provider="openai",
            model_name="gpt-4",
            display_name="GPT-4",
            description="Test model",
            max_input_tokens=8192,
            max_output_tokens=4096,
            dollars_per_million_tokens_input=None,  # No pricing
            dollars_per_million_tokens_output=None,
            is_active=True,
        )

        response = LLMResponse(
            content="Hello",
            model="openai:gpt-4",
            provider="openai",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

        cost = await calculator.calculate_cost(response, registry_model)
        assert cost is None

    @pytest.mark.asyncio
    async def test_calculate_cost_fetches_from_registry(self, calculator, sample_response):
        """Should fetch from registry when model not provided."""
        registry_model = RegistryModel(
            provider="openai",
            model_name="gpt-4",
            display_name="GPT-4",
            description="Test model",
            max_input_tokens=8192,
            max_output_tokens=4096,
            dollars_per_million_tokens_input=10.0,
            dollars_per_million_tokens_output=20.0,
            is_active=True,
        )

        calculator.registry.fetch_current_models = AsyncMock(return_value=[registry_model])

        cost = await calculator.calculate_cost(sample_response)

        assert cost is not None
        assert cost["total_cost"] == pytest.approx(
            0.002
        )  # 100*10 + 50*20 = 1000+1000 = 2000 per million = 0.002
        calculator.registry.fetch_current_models.assert_called_once_with("openai")

    @pytest.mark.asyncio
    async def test_calculate_cost_model_not_in_registry(self, calculator, sample_response):
        """Should return None when model not found in registry."""
        # Return empty list
        calculator.registry.fetch_current_models = AsyncMock(return_value=[])

        cost = await calculator.calculate_cost(sample_response)
        assert cost is None

    @pytest.mark.asyncio
    async def test_calculate_cost_registry_fetch_fails(self, calculator, sample_response):
        """Should handle registry fetch failure gracefully."""
        calculator.registry.fetch_current_models = AsyncMock(
            side_effect=Exception("Registry error")
        )

        cost = await calculator.calculate_cost(sample_response)
        assert cost is None

    @pytest.mark.asyncio
    async def test_calculate_cost_zero_tokens(self, calculator, sample_registry_model):
        """Should handle zero tokens correctly."""
        response = LLMResponse(
            content="Hello",
            model="openai:gpt-4",
            provider="openai",
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )

        cost = await calculator.calculate_cost(response, sample_registry_model)

        assert cost is not None
        assert cost["input_cost"] == 0.0
        assert cost["output_cost"] == 0.0
        assert cost["total_cost"] == 0.0

    @pytest.mark.asyncio
    async def test_calculate_cost_large_numbers(self, calculator, sample_registry_model):
        """Should handle large token counts correctly."""
        response = LLMResponse(
            content="Hello",
            model="openai:gpt-4",
            provider="openai",
            usage={
                "prompt_tokens": 1_000_000,  # 1 million tokens
                "completion_tokens": 500_000,  # 0.5 million tokens
            },
        )

        cost = await calculator.calculate_cost(response, sample_registry_model)

        assert cost is not None
        # 1M * $30/1M = $30
        assert cost["input_cost"] == pytest.approx(30.0)
        # 0.5M * $60/1M = $30
        assert cost["output_cost"] == pytest.approx(30.0)
        # Total = $60
        assert cost["total_cost"] == pytest.approx(60.0)

    def test_calculate_token_cost(self, calculator):
        """Should calculate token cost correctly."""
        # 100 tokens at $30 per million
        cost = calculator._calculate_token_cost(100, 30.0)
        assert cost == pytest.approx(0.003)

        # 1 million tokens at $30 per million
        cost = calculator._calculate_token_cost(1_000_000, 30.0)
        assert cost == pytest.approx(30.0)

        # 0 tokens
        cost = calculator._calculate_token_cost(0, 30.0)
        assert cost == 0.0

    def test_add_cost_to_response(self, calculator, sample_response):
        """Should add cost info to response."""
        cost_info = {
            "input_cost": 0.003,
            "output_cost": 0.003,
            "total_cost": 0.006,
        }

        calculator.add_cost_to_response(sample_response, cost_info)

        assert sample_response.usage["cost"] == 0.006
        assert sample_response.usage["cost_breakdown"]["input"] == 0.003
        assert sample_response.usage["cost_breakdown"]["output"] == 0.003

    def test_add_cost_to_response_no_usage(self, calculator):
        """Should create usage dict if not present."""
        response = LLMResponse(
            content="Hello",
            model="openai:gpt-4",
            provider="openai",
        )

        cost_info = {
            "input_cost": 0.003,
            "output_cost": 0.003,
            "total_cost": 0.006,
        }

        calculator.add_cost_to_response(response, cost_info)

        assert response.usage is not None
        assert response.usage["cost"] == 0.006

    def test_get_zero_cost_info(self, calculator):
        """Should return zero cost dict."""
        zero_cost = calculator.get_zero_cost_info()

        assert zero_cost["input_cost"] == 0.0
        assert zero_cost["output_cost"] == 0.0
        assert zero_cost["total_cost"] == 0.0
