"""Unit tests for registry URL validation policy."""

import pytest

from llmring.validation import InputValidator


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/llmring-registry",
        "https://llmring.github.io/registry",
        "http://example.com/registry",
        "http://invalid-registry-url.example.com",
        "http://localhost:8000/registry",
        "https://localhost:8443/registry",
        "http://127.0.0.1:8000/registry",
        "https://127.0.0.1:8443/registry",
        "http://[::1]:8000/registry",
        "https://[::1]:8443/registry",
    ],
)
def test_validate_registry_url_allows_expected_urls(url: str) -> None:
    """Accept expected safe/dev registry URLs."""
    InputValidator.validate_registry_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://llmring.github.io/registry",
        "ftp://example.com/registry",
        "https://0.0.0.0/registry",
        "https://[::]/registry",
        "https:///registry",
        "",
    ],
)
def test_validate_registry_url_rejects_unsafe_or_invalid_urls(url: str) -> None:
    """Reject invalid schemes/hosts and non-https remote URLs."""
    with pytest.raises(ValueError):
        InputValidator.validate_registry_url(url)
