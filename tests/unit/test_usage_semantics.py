"""Cross-provider usage-token semantics.

Anthropic and OpenAI disagree about what their "input tokens" number means:

  * Anthropic  ``usage.input_tokens``  EXCLUDES cache reads and cache writes.
  * OpenAI     ``usage.prompt_tokens`` INCLUDES cached reads.

The cost calculator assumes the OpenAI convention (it subtracts cache reads to
find the tokens billed at the base rate). These tests pin both halves of the
normalisation so the two can never silently drift apart again.
"""

import pytest

from llmring.providers.anthropic_api import build_usage_dict
from llmring.providers.openai_api import OpenAIProvider
from llmring.registry import RegistryModel
from llmring.schemas import LLMResponse
from llmring.services.cost_calculator import CostCalculator


class _Obj:
    """Minimal stand-in for an SDK usage object."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


SONNET = RegistryModel(
    provider="anthropic",
    model_name="claude-sonnet-4-5-20250929",
    display_name="Sonnet 4.5",
    dollars_per_million_tokens_input=3.0,
    dollars_per_million_tokens_output=15.0,
    dollars_per_million_tokens_cache_read=0.30,
    dollars_per_million_tokens_cache_write_5m=3.75,
)

GPT5_MINI = RegistryModel(
    provider="openai",
    model_name="gpt-5-mini",
    display_name="gpt-5-mini",
    dollars_per_million_tokens_input=0.25,
    dollars_per_million_tokens_output=2.0,
    dollars_per_million_tokens_cached_input=0.025,
)


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------

def test_anthropic_prompt_tokens_include_cache_reads():
    """input_tokens excludes cache reads, so we must add them back."""
    usage = build_usage_dict(
        _Obj(input_tokens=1500, output_tokens=150, cache_read_input_tokens=8000,
             cache_creation_input_tokens=0)
    )
    assert usage["prompt_tokens"] == 9500          # 1500 fresh + 8000 cache reads
    assert usage["cache_read_input_tokens"] == 8000
    assert usage["input_tokens_uncached"] == 1500  # provider's raw figure preserved


def test_anthropic_cache_writes_are_not_folded_into_prompt_tokens():
    """Cache-creation tokens bill at the write rate; counting them in
    prompt_tokens too would bill them twice."""
    usage = build_usage_dict(
        _Obj(input_tokens=1000, output_tokens=50, cache_read_input_tokens=0,
             cache_creation_input_tokens=4000)
    )
    assert usage["prompt_tokens"] == 1000
    assert usage["cache_creation_input_tokens"] == 4000
    assert usage["total_tokens"] == 1000 + 4000 + 50


@pytest.mark.asyncio
async def test_anthropic_cached_call_costs_the_published_amount():
    usage = build_usage_dict(
        _Obj(input_tokens=1500, output_tokens=150, cache_read_input_tokens=8000,
             cache_creation_input_tokens=0)
    )
    got = await CostCalculator(registry=None).calculate_cost(
        LLMResponse(content="", model="anthropic:claude-sonnet-4-5-20250929", usage=usage),
        registry_model=SONNET,
    )
    expected = 1500 / 1e6 * 3.0 + 8000 / 1e6 * 0.30 + 150 / 1e6 * 15.0
    assert got["total_cost"] == pytest.approx(expected, rel=1e-9)
    # The fresh tokens must actually be charged - the old bug billed them at zero.
    assert got["input_cost"] == pytest.approx(1500 / 1e6 * 3.0, rel=1e-9)


@pytest.mark.asyncio
async def test_anthropic_cache_write_billed_at_write_rate_not_twice():
    usage = build_usage_dict(
        _Obj(input_tokens=1000, output_tokens=50, cache_read_input_tokens=0,
             cache_creation_input_tokens=4000,
             cache_creation=_Obj(ephemeral_5m_input_tokens=4000,
                                 ephemeral_1h_input_tokens=0))
    )
    got = await CostCalculator(registry=None).calculate_cost(
        LLMResponse(content="", model="anthropic:claude-sonnet-4-5-20250929", usage=usage),
        registry_model=SONNET,
    )
    expected = 1000 / 1e6 * 3.0 + 4000 / 1e6 * 3.75 + 50 / 1e6 * 15.0
    assert got["total_cost"] == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------

def test_openai_cached_tokens_are_surfaced():
    mapped = OpenAIProvider._map_responses_usage(
        None,
        _Obj(prompt_tokens=9500, completion_tokens=150, total_tokens=9650,
             prompt_tokens_details=_Obj(cached_tokens=8000)),
    )
    assert mapped["cached_tokens"] == 8000
    # OpenAI already counts cached tokens inside prompt_tokens - do not add them again.
    assert mapped["prompt_tokens"] == 9500


def test_openai_reasoning_tokens_are_surfaced():
    mapped = OpenAIProvider._map_responses_usage(
        None,
        _Obj(prompt_tokens=100, completion_tokens=900, total_tokens=1000,
             completion_tokens_details=_Obj(reasoning_tokens=800)),
    )
    assert mapped["reasoning_tokens"] == 800


def test_openai_usage_without_details_still_maps():
    mapped = OpenAIProvider._map_responses_usage(
        None, _Obj(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    assert mapped == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


@pytest.mark.asyncio
async def test_openai_cached_call_costs_the_published_amount():
    mapped = OpenAIProvider._map_responses_usage(
        None,
        _Obj(prompt_tokens=9500, completion_tokens=150, total_tokens=9650,
             prompt_tokens_details=_Obj(cached_tokens=8000)),
    )
    got = await CostCalculator(registry=None).calculate_cost(
        LLMResponse(content="", model="openai:gpt-5-mini", usage=mapped),
        registry_model=GPT5_MINI,
    )
    expected = 1500 / 1e6 * 0.25 + 8000 / 1e6 * 0.025 + 150 / 1e6 * 2.0
    assert got["total_cost"] == pytest.approx(expected, rel=1e-9)


@pytest.mark.asyncio
async def test_the_two_providers_agree_on_an_identical_billing_situation():
    """Same real-world call shape, same rates, expressed in each provider's
    native convention, must produce the same cost. This is the invariant the
    old code violated in opposite directions."""
    anthropic_usage = build_usage_dict(
        _Obj(input_tokens=1500, output_tokens=150, cache_read_input_tokens=8000,
             cache_creation_input_tokens=0)
    )
    openai_usage = OpenAIProvider._map_responses_usage(
        None,
        _Obj(prompt_tokens=9500, completion_tokens=150, total_tokens=9650,
             prompt_tokens_details=_Obj(cached_tokens=8000)),
    )
    assert anthropic_usage["prompt_tokens"] == openai_usage["prompt_tokens"] == 9500

    same_rates = dict(
        dollars_per_million_tokens_input=3.0,
        dollars_per_million_tokens_output=15.0,
        dollars_per_million_tokens_cache_read=0.30,
        dollars_per_million_tokens_cached_input=0.30,
    )
    a_model = RegistryModel(provider="anthropic", model_name="m", display_name="m", **same_rates)
    o_model = RegistryModel(provider="openai", model_name="m", display_name="m", **same_rates)
    calc = CostCalculator(registry=None)
    a = await calc.calculate_cost(
        LLMResponse(content="", model="anthropic:m", usage=anthropic_usage), registry_model=a_model)
    o = await calc.calculate_cost(
        LLMResponse(content="", model="openai:m", usage=openai_usage), registry_model=o_model)
    assert a["total_cost"] == pytest.approx(o["total_cost"], rel=1e-9)
