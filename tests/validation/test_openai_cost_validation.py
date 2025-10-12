"""
Validation test for OpenAI cost calculations.

This test validates that the cost calculation logic in CostCalculator
correctly uses the pricing data from the OpenAI registry.
"""

import pytest

from llmring.registry import RegistryClient, RegistryModel
from llmring.schemas import LLMResponse
from llmring.services.cost_calculator import CostCalculator


class TestOpenAICostValidation:
    """Validate OpenAI cost calculations against registry data."""

    @pytest.fixture
    async def registry_client(self):
        """Create a real registry client."""
        return RegistryClient()

    @pytest.fixture
    async def calculator(self, registry_client):
        """Create a CostCalculator with real registry."""
        return CostCalculator(registry_client)

    @pytest.fixture
    async def openai_models(self, registry_client):
        """Fetch current OpenAI models from registry."""
        return await registry_client.fetch_current_models("openai")

    @pytest.mark.asyncio
    async def test_gpt_4o_mini_cost_calculation(self, calculator, openai_models):
        """Validate cost calculation for gpt-4o-mini."""
        # Find gpt-4o-mini in registry
        model = next((m for m in openai_models if m.model_name == "gpt-4o-mini"), None)
        assert model is not None, "gpt-4o-mini not found in registry"

        # Create a response with known token counts
        response = LLMResponse(
            content="Test response",
            model="openai:gpt-4o-mini",
            provider="openai",
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
            },
        )

        # Calculate cost
        cost = await calculator.calculate_cost(response, model)

        assert cost is not None
        assert cost["cost_per_million_input"] == model.dollars_per_million_tokens_input
        assert cost["cost_per_million_output"] == model.dollars_per_million_tokens_output

        # Manually calculate expected costs
        expected_input_cost = (1000 / 1_000_000) * model.dollars_per_million_tokens_input
        expected_output_cost = (500 / 1_000_000) * model.dollars_per_million_tokens_output
        expected_total_cost = expected_input_cost + expected_output_cost

        assert cost["input_cost"] == pytest.approx(expected_input_cost)
        assert cost["output_cost"] == pytest.approx(expected_output_cost)
        assert cost["total_cost"] == pytest.approx(expected_total_cost)

        # Verify the formula
        assert cost["input_cost"] == pytest.approx(
            (response.usage["prompt_tokens"] / 1_000_000) * cost["cost_per_million_input"]
        )
        assert cost["output_cost"] == pytest.approx(
            (response.usage["completion_tokens"] / 1_000_000) * cost["cost_per_million_output"]
        )

    @pytest.mark.asyncio
    async def test_gpt_4_1_cost_calculation(self, calculator, openai_models):
        """Validate cost calculation for gpt-4.1."""
        # Find gpt-4.1 in registry
        model = next((m for m in openai_models if m.model_name == "gpt-4.1"), None)
        assert model is not None, "gpt-4.1 not found in registry"

        # Create a response with known token counts
        response = LLMResponse(
            content="Test response",
            model="openai:gpt-4.1",
            provider="openai",
            usage={
                "prompt_tokens": 10000,
                "completion_tokens": 2000,
                "total_tokens": 12000,
            },
        )

        # Calculate cost
        cost = await calculator.calculate_cost(response, model)

        assert cost is not None

        # Manually calculate expected costs
        expected_input_cost = (10000 / 1_000_000) * model.dollars_per_million_tokens_input
        expected_output_cost = (2000 / 1_000_000) * model.dollars_per_million_tokens_output
        expected_total_cost = expected_input_cost + expected_output_cost

        assert cost["input_cost"] == pytest.approx(expected_input_cost)
        assert cost["output_cost"] == pytest.approx(expected_output_cost)
        assert cost["total_cost"] == pytest.approx(expected_total_cost)

    @pytest.mark.asyncio
    async def test_o1_pro_cost_calculation(self, calculator, openai_models):
        """Validate cost calculation for o1-pro (expensive model)."""
        # Find o1-pro in registry
        model = next((m for m in openai_models if m.model_name == "o1-pro"), None)
        assert model is not None, "o1-pro not found in registry"

        # Create a response with known token counts
        response = LLMResponse(
            content="Test response",
            model="openai:o1-pro",
            provider="openai",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        )

        # Calculate cost
        cost = await calculator.calculate_cost(response, model)

        assert cost is not None

        # o1-pro should have high pricing
        assert model.dollars_per_million_tokens_input >= 100.0  # At least $100/M
        assert model.dollars_per_million_tokens_output >= 400.0  # At least $400/M

        # Manually calculate expected costs
        expected_input_cost = (100 / 1_000_000) * model.dollars_per_million_tokens_input
        expected_output_cost = (50 / 1_000_000) * model.dollars_per_million_tokens_output
        expected_total_cost = expected_input_cost + expected_output_cost

        assert cost["input_cost"] == pytest.approx(expected_input_cost)
        assert cost["output_cost"] == pytest.approx(expected_output_cost)
        assert cost["total_cost"] == pytest.approx(expected_total_cost)

    @pytest.mark.asyncio
    async def test_all_active_models_have_pricing(self, openai_models):
        """Validate that all active OpenAI models have pricing information."""
        active_models = [m for m in openai_models if m.is_active]

        for model in active_models:
            assert (
                model.dollars_per_million_tokens_input is not None
            ), f"{model.model_name} missing input pricing"
            assert (
                model.dollars_per_million_tokens_output is not None
            ), f"{model.model_name} missing output pricing"
            assert (
                model.dollars_per_million_tokens_input >= 0
            ), f"{model.model_name} has negative input pricing"
            assert (
                model.dollars_per_million_tokens_output >= 0
            ), f"{model.model_name} has negative output pricing"

    @pytest.mark.asyncio
    async def test_cost_formula_consistency(self, calculator):
        """Validate that the cost calculation formula is correct."""
        # Create a simple model with known pricing
        test_model = RegistryModel(
            provider="openai",
            model_name="test-model",
            display_name="Test Model",
            description="Test",
            max_input_tokens=1000,
            max_output_tokens=1000,
            dollars_per_million_tokens_input=10.0,  # $10 per million input tokens
            dollars_per_million_tokens_output=20.0,  # $20 per million output tokens
            is_active=True,
        )

        # Test with 100,000 input tokens and 50,000 output tokens
        response = LLMResponse(
            content="Test",
            model="openai:test-model",
            provider="openai",
            usage={
                "prompt_tokens": 100_000,
                "completion_tokens": 50_000,
                "total_tokens": 150_000,
            },
        )

        cost = await calculator.calculate_cost(response, test_model)

        # Expected: (100,000 / 1,000,000) * 10 = 1.0
        assert cost["input_cost"] == pytest.approx(1.0)
        # Expected: (50,000 / 1,000,000) * 20 = 1.0
        assert cost["output_cost"] == pytest.approx(1.0)
        # Expected: 1.0 + 1.0 = 2.0
        assert cost["total_cost"] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_fractional_cents_precision(self, calculator):
        """Validate precision for small token counts (fractions of a cent)."""
        # Create a model with typical pricing
        test_model = RegistryModel(
            provider="openai",
            model_name="test-model",
            display_name="Test Model",
            description="Test",
            max_input_tokens=1000,
            max_output_tokens=1000,
            dollars_per_million_tokens_input=0.15,  # $0.15 per million
            dollars_per_million_tokens_output=0.6,  # $0.60 per million
            is_active=True,
        )

        # Test with small token counts
        response = LLMResponse(
            content="Test",
            model="openai:test-model",
            provider="openai",
            usage={
                "prompt_tokens": 100,  # Should be $0.000015
                "completion_tokens": 50,  # Should be $0.000030
                "total_tokens": 150,
            },
        )

        cost = await calculator.calculate_cost(response, test_model)

        # Expected: (100 / 1,000,000) * 0.15 = 0.000015
        assert cost["input_cost"] == pytest.approx(0.000015)
        # Expected: (50 / 1,000,000) * 0.6 = 0.000030
        assert cost["output_cost"] == pytest.approx(0.000030)
        # Expected: 0.000015 + 0.000030 = 0.000045
        assert cost["total_cost"] == pytest.approx(0.000045)

    @pytest.mark.asyncio
    async def test_registry_lookup_integration(self, calculator):
        """Test cost calculation with automatic registry lookup."""
        # Create a response without providing registry model
        response = LLMResponse(
            content="Test response",
            model="openai:gpt-4o-mini",
            provider="openai",
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
            },
        )

        # Calculate cost - should fetch from registry automatically
        cost = await calculator.calculate_cost(response)

        assert cost is not None
        assert cost["total_cost"] > 0
        assert cost["cost_per_million_input"] > 0
        assert cost["cost_per_million_output"] > 0

        # Verify the formula matches
        expected_input = (1000 / 1_000_000) * cost["cost_per_million_input"]
        expected_output = (500 / 1_000_000) * cost["cost_per_million_output"]

        assert cost["input_cost"] == pytest.approx(expected_input)
        assert cost["output_cost"] == pytest.approx(expected_output)
        assert cost["total_cost"] == pytest.approx(expected_input + expected_output)
