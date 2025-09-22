"""Integration tests for optional registry behavior."""

from pathlib import Path

import pytest

from llmring import LLMRing
from llmring.schemas import LLMRequest


@pytest.mark.asyncio
@pytest.mark.integration
async def test_models_work_without_registry():
    """Test that models work even when not in registry."""

    # Use an invalid registry URL to ensure registry is unavailable
    test_lockfile = Path(__file__).parent.parent / "llmring.lock.json"
    ring = LLMRing(
        registry_url="http://invalid-registry-url.example.com",
        lockfile_path=str(test_lockfile),
    )

    # Test with a real model that might not be in registry
    request = LLMRequest(
        model="openai:gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply with just 'OK'"}],
        max_tokens=10,
    )

    try:
        response = await ring.chat(request)
        # If we get here, the model worked despite potentially not being in registry
        assert response.content
        assert response.model
    except Exception as e:
        # The only acceptable failure is from the API itself (e.g., invalid API key)
        assert "API" in str(e) or "authentication" in str(e).lower() or "401" in str(e)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_registry_validation_is_advisory_not_mandatory():
    """Test that registry validation doesn't block model usage."""

    # Use test registry
    from pathlib import Path

    test_registry = Path(__file__).parent.parent / "resources" / "registry"
    test_lockfile = Path(__file__).parent.parent / "llmring.lock.json"
    ring = LLMRing(registry_url=f"file://{test_registry}", lockfile_path=str(test_lockfile))

    # Try a model that exists in the test registry
    request1 = LLMRequest(
        model="openai:gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Reply with just 'OK'"}],
        max_tokens=10,
    )

    try:
        response1 = await ring.chat(request1)
        assert response1.content
    except Exception as e:
        # API errors are acceptable
        assert "API" in str(e) or "authentication" in str(e).lower()

    # Try a model that doesn't exist in the test registry
    # This should still work (with a warning) as validation is advisory
    request2 = LLMRequest(
        model="openai:gpt-4o-mini",  # Not in test registry
        messages=[{"role": "user", "content": "Reply with just 'OK'"}],
        max_tokens=10,
    )

    try:
        response2 = await ring.chat(request2)
        assert response2.content
    except Exception as e:
        # Should only fail due to API errors, not registry validation
        assert "API" in str(e) or "authentication" in str(e).lower() or "404" in str(e)
