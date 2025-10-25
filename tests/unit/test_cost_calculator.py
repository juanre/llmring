import pytest

from llmring.registry import RegistryModel
from llmring.schemas import LLMResponse
from llmring.services.cost_calculator import CostCalculator


@pytest.mark.asyncio
async def test_cost_calculator_handles_cached_tokens():
    """CostCalculator should account for cached read/write tokens."""
    registry_model = RegistryModel(
        provider="openai",
        model_name="test-cache",
        display_name="OpenAI Cache Test",
        description="Synthetic model covering extended pricing fields",
        dollars_per_million_tokens_input=1.0,
        dollars_per_million_tokens_output=2.0,
        dollars_per_million_tokens_cached_input=0.25,
        dollars_per_million_tokens_cache_write_5m=0.5,
        dollars_per_million_tokens_cache_write_1h=0.75,
        dollars_per_million_tokens_cache_read=0.1,
        supports_caching=True,
        supports_streaming=True,
        is_active=True,
    )

    response = LLMResponse(
        content="ok",
        model="openai:test-cache",
        usage={
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "cached_tokens": 400,
            "cache_creation_input_tokens": 50,
        },
    )

    calculator = CostCalculator(registry=None)  # registry not needed when passing model explicitly
    cost_info = await calculator.calculate_cost(response, registry_model=registry_model)

    assert cost_info is not None
    # Non-cached prompt tokens = 1000 - 400 = 600 → $0.0006
    assert pytest.approx(cost_info["input_cost"], rel=1e-6) == 600 / 1_000_000
    # Cached reads 400 tokens at $0.1 → $0.00004
    assert pytest.approx(cost_info["cache_read_cost"], rel=1e-6) == 400 / 1_000_000 * 0.1
    # Cache write (generic) 50 tokens at fallback 0.5 → $0.000025
    assert pytest.approx(cost_info["cache_write_cost_other"], rel=1e-6) == 50 / 1_000_000 * 0.5
    # Output tokens 200 at $2 → $0.0004
    assert pytest.approx(cost_info["output_cost"], rel=1e-6) == 200 / 1_000_000 * 2.0
    expected_total = (
        cost_info["input_cost"]
        + cost_info["cache_read_cost"]
        + cost_info["cache_write_cost_other"]
        + cost_info["output_cost"]
    )
    assert pytest.approx(cost_info["total_cost"], rel=1e-9) == expected_total


@pytest.mark.asyncio
async def test_cost_calculator_anthropic_cache_buckets():
    """Anthropic cache buckets (5m / 1h) should be billed separately."""
    registry_model = RegistryModel(
        provider="anthropic",
        model_name="test-model",
        display_name="Anthropic Test Model",
        description="Synthetic model with cache pricing",
        dollars_per_million_tokens_input=2.0,
        dollars_per_million_tokens_output=6.0,
        dollars_per_million_tokens_cache_write_5m=3.0,
        dollars_per_million_tokens_cache_write_1h=5.0,
        dollars_per_million_tokens_cache_read=0.2,
        supports_caching=True,
        supports_streaming=True,
        is_active=True,
    )

    response = LLMResponse(
        content="ok",
        model="anthropic:test-model",
        usage={
            "prompt_tokens": 5000,
            "completion_tokens": 1000,
            "cache_read_input_tokens": 1200,
            "cache_creation_5m_tokens": 800,
            "cache_creation_1h_tokens": 600,
        },
    )

    calculator = CostCalculator(registry=None)
    cost_info = await calculator.calculate_cost(response, registry_model=registry_model)

    assert cost_info is not None
    non_cached_prompt = 5000 - 1200
    assert pytest.approx(cost_info["input_cost"], rel=1e-6) == non_cached_prompt / 1_000_000 * 2.0
    assert pytest.approx(cost_info["cache_read_cost"], rel=1e-6) == 1200 / 1_000_000 * 0.2
    assert pytest.approx(cost_info["cache_write_cost_5m"], rel=1e-6) == 800 / 1_000_000 * 3.0
    assert pytest.approx(cost_info["cache_write_cost_1h"], rel=1e-6) == 600 / 1_000_000 * 5.0
    assert pytest.approx(cost_info["output_cost"], rel=1e-6) == 1000 / 1_000_000 * 6.0


@pytest.mark.asyncio
async def test_cost_calculator_reasoning_tokens():
    """Reasoning tokens should be charged using thinking rate."""
    registry_model = RegistryModel(
        provider="openai",
        model_name="reasoning-test",
        display_name="Reasoning Test",
        description="Synthetic reasoning model",
        dollars_per_million_tokens_input=1.0,
        dollars_per_million_tokens_output=1.5,
        dollars_per_million_tokens_output_thinking=4.0,
        supports_thinking=True,
        is_reasoning_model=True,
        min_recommended_reasoning_tokens=2000,
        is_active=True,
    )

    response = LLMResponse(
        content="ok",
        model="openai:reasoning-test",
        usage={
            "prompt_tokens": 1000,
            "completion_tokens": 300,
            "reasoning_tokens": 120,
        },
    )

    calculator = CostCalculator(registry=None)
    cost_info = await calculator.calculate_cost(response, registry_model=registry_model)

    assert cost_info is not None
    # Prompt cost
    assert pytest.approx(cost_info["input_cost"], rel=1e-6) == 1000 / 1_000_000 * 1.0
    # Reasoning cost at dedicated rate
    assert pytest.approx(cost_info["reasoning_cost"], rel=1e-6) == 120 / 1_000_000 * 4.0
    # Completion cost should only include remaining tokens (300 - 120 = 180)
    assert pytest.approx(cost_info["output_cost"], rel=1e-6) == 180 / 1_000_000 * 1.5
