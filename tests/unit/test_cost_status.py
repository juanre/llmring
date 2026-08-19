"""Unpriced calls must be a visible, typed fact - never a silent zero.

Before this, a model missing from the registry produced cost=None, a DEBUG log,
and a usage dict with no cost key. A batch run could therefore record thousands
of calls with no cost and nothing visible at normal log levels.
"""

import logging

import pytest

from llmring.exceptions import CostTrackingError
from llmring.registry import RegistryModel
from llmring.schemas import LLMRequest, LLMResponse, Message
from llmring.service import LLMRing
from llmring.services.cost_calculator import (
    COST_STATUS_MODEL_NOT_IN_REGISTRY,
    COST_STATUS_NO_PRICING_IN_REGISTRY,
    COST_STATUS_NO_USAGE,
    COST_STATUS_PRICED,
    CostCalculator,
)

USAGE = {"prompt_tokens": 2000, "completion_tokens": 150, "total_tokens": 2150}

PRICED = RegistryModel(
    provider="anthropic",
    model_name="m",
    display_name="m",
    dollars_per_million_tokens_input=3.0,
    dollars_per_million_tokens_output=15.0,
)
UNPRICED = RegistryModel(
    provider="anthropic",
    model_name="m",
    display_name="m",
    dollars_per_million_tokens_input=None,
    dollars_per_million_tokens_output=None,
)


class _EmptyRegistry:
    async def fetch_current_models(self, provider):
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage,registry_model,registry,expected_status,expect_cost",
    [
        (USAGE, PRICED, None, COST_STATUS_PRICED, True),
        (None, PRICED, None, COST_STATUS_NO_USAGE, False),
        (USAGE, UNPRICED, None, COST_STATUS_NO_PRICING_IN_REGISTRY, False),
        (USAGE, None, _EmptyRegistry(), COST_STATUS_MODEL_NOT_IN_REGISTRY, False),
    ],
)
async def test_status_reports_the_reason(
    usage, registry_model, registry, expected_status, expect_cost
):
    calc = CostCalculator(registry=registry)
    resp = LLMResponse(content="", model="anthropic:m", usage=usage)
    cost, status = await calc.calculate_cost_detailed(resp, registry_model)
    assert status == expected_status
    assert (cost is not None) is expect_cost


@pytest.mark.asyncio
async def test_invalid_model_format_is_its_own_status():
    calc = CostCalculator(registry=None)
    resp = LLMResponse(content="", model="no-provider-prefix", usage=dict(USAGE))
    cost, status = await calc.calculate_cost_detailed(resp)
    assert status == "invalid_model_format"
    assert cost is None


@pytest.mark.asyncio
async def test_calculate_cost_still_returns_just_the_cost():
    """The old single-value API keeps working for existing callers."""
    calc = CostCalculator(registry=None)
    resp = LLMResponse(content="", model="anthropic:m", usage=dict(USAGE))
    assert (await calc.calculate_cost(resp, PRICED))["total_cost"] > 0


# --------------------------------------------------------------------------
# strict mode, through the real chat() path
# --------------------------------------------------------------------------


class _StubProvider:
    """Returns a fixed response with usage but for a model nobody prices."""

    def __init__(self, model="ghost-model-9000"):
        self.model = model

    async def chat(self, **kwargs):
        return LLMResponse(content="ok", model=self.model, usage=dict(USAGE), finish_reason="stop")

    async def get_default_model(self):
        return self.model

    async def aclose(self):
        return None


def _ring_with_stub(**kw):
    ring = LLMRing(**kw)
    ring.providers = {"stub": _StubProvider()}
    return ring


@pytest.mark.asyncio
async def test_unpriced_call_records_status_and_warns(caplog):
    ring = _ring_with_stub()
    req = LLMRequest(model="stub:ghost-model-9000", messages=[Message(role="user", content="hi")])
    with caplog.at_level(logging.WARNING):
        resp = await ring.chat(req)
    assert resp.usage["cost_status"] == COST_STATUS_MODEL_NOT_IN_REGISTRY
    assert "cost" not in resp.usage
    # The whole point: visible at WARNING, not buried at DEBUG.
    assert any("No cost recorded" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_strict_mode_raises_instead_of_recording_a_silent_zero():
    ring = _ring_with_stub(strict_cost=True)
    req = LLMRequest(model="stub:ghost-model-9000", messages=[Message(role="user", content="hi")])
    with pytest.raises(CostTrackingError) as exc:
        await ring.chat(req)
    assert exc.value.cost_status == COST_STATUS_MODEL_NOT_IN_REGISTRY
    assert "ghost-model-9000" in exc.value.model


@pytest.mark.asyncio
async def test_strict_mode_reads_the_environment(monkeypatch):
    monkeypatch.setenv("LLMRING_STRICT_COST", "1")
    assert LLMRing().strict_cost is True
