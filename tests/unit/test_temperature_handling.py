"""Temperature must never be invented, and a 400 about it is not a missing model.

Regression guard for the failure minerva's induced-outage test exposed:
claude-sonnet-5 rejects any request carrying `temperature` with HTTP 400
"`temperature` is deprecated for this model". Three separate defects combined
to make that fatal, and each is pinned below.
"""

import pytest

from llmring.providers.anthropic_api import AnthropicProvider
from llmring.providers.error_handler import ProviderErrorHandler
from llmring.schemas import Message


class _Captured(BaseException):
    """Abort the call once we have the request params.

    Deliberately a BaseException: the provider wraps SDK failures in
    `except Exception`, which would otherwise swallow the probe and turn it
    into a provider error instead of returning the params.
    """

    def __init__(self, params):
        self.params = params


def _provider_capturing_params():
    # Build the provider normally (so the circuit breaker and error handler
    # exist) with a dummy key, then swap only the SDK client.
    p = AnthropicProvider(api_key="sk-ant-dummy-for-unit-test")

    class _Messages:
        async def create(self, **params):
            raise _Captured(params)

    class _Client:
        messages = _Messages()

    p.client = _Client()
    return p


async def _params_for(temperature):
    """Run _chat_non_streaming far enough to capture what would be sent."""
    p = _provider_capturing_params()
    try:
        await p._chat_non_streaming(
            messages=[Message(role="user", content="hi")],
            model="claude-sonnet-5",
            temperature=temperature,
            max_tokens=16,
            response_format=None,
            tools=None,
            tool_choice=None,
            extra_params=None,
            timeout=None,
        )
    except _Captured as c:
        return c.params
    raise AssertionError("SDK call was not reached")


@pytest.mark.asyncio
async def test_none_temperature_sends_no_temperature_at_all():
    """The bug: `temperature or 0.7` invented 0.7 for callers passing None,
    re-injecting the very parameter the service layer had just stripped."""
    params = await _params_for(None)
    assert "temperature" not in params


@pytest.mark.asyncio
async def test_zero_temperature_is_preserved_not_rewritten_to_default():
    """Separate bug in the same expression: 0.0 is falsy, so a deliberate
    request for deterministic output silently became 0.7."""
    params = await _params_for(0.0)
    assert params["temperature"] == 0.0


@pytest.mark.asyncio
async def test_explicit_temperature_is_passed_through():
    params = await _params_for(0.2)
    assert params["temperature"] == 0.2


# --------------------------------------------------------------------------
# error classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "message,expected_unknown_model",
    [
        # The exact 400 that misdetected, sending a reviewer after a
        # nonexistent model problem during an outage test.
        ("invalid_request_error: `temperature` is deprecated for this model.", False),
        ("invalid_request_error: unsupported parameter: 'top_k' for this model", False),
        ("messages: at least one message is required", False),
        # Genuine unknown-model signals stay detected.
        ("not_found_error: model: claude-mythos-5", True),
        ("Unknown model: claude-imaginary-9", True),
        ("model not found", True),
        ("model not supported", True),
        ("unsupported model: claude-imaginary-9", True),
        ("invalid model", True),
    ],
)
def test_only_real_missing_model_errors_are_classified_as_such(
    message, expected_unknown_model
):
    assert ProviderErrorHandler._mentions_unknown_model(message) is expected_unknown_model
