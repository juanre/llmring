"""Encoding selection for OpenAI token counting.

Regression guard: unknown/current OpenAI models must not fall back to
cl100k_base. Every model from gpt-4o onward (gpt-4.1, gpt-5*, o1/o3/o4)
uses o200k_base; only the original gpt-4 family and gpt-3.5 use cl100k_base.
"""

import pytest

tiktoken = pytest.importorskip("tiktoken")

from llmring.token_counter import count_tokens_openai

# Long enough that the two encodings disagree on the token count.
TEXT = (
    "Reconcile the 2026 invoice totals against the ledger, "
    "flagging any discrepancy above 0.5% for manual review."
)
MESSAGES = [{"role": "user", "content": TEXT}]


def _expected(encoding_name: str) -> int:
    enc = tiktoken.get_encoding(encoding_name)
    # Mirrors count_tokens_openai's per-message accounting.
    return 4 + len(enc.encode("user")) + len(enc.encode(TEXT)) + 2


@pytest.mark.parametrize(
    "model",
    ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-nano", "o3-mini", "gpt-4o-mini"],
)
def test_current_models_use_o200k(model):
    assert count_tokens_openai(MESSAGES, model) == _expected("o200k_base")


@pytest.mark.parametrize("model", ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"])
def test_legacy_models_use_cl100k(model):
    assert count_tokens_openai(MESSAGES, model) == _expected("cl100k_base")


def test_the_two_encodings_actually_differ_on_this_text():
    """Guards the test itself: if the encodings agreed, the tests above
    would pass regardless of which one the code picked."""
    assert _expected("o200k_base") != _expected("cl100k_base")
