"""
Common fixtures for integration tests.
"""

import os

import pytest

from llmring.service import LLMRing


@pytest.fixture
async def service():
    """Create LLMRing service with test lockfile."""
    test_lockfile = os.path.join(os.path.dirname(os.path.dirname(__file__)), "llmring.lock.json")
    service = LLMRing(lockfile_path=test_lockfile)
    yield service
    # Clean up resources
    await service.close()


@pytest.fixture
def llmring(service):
    """Alias for service fixture for backwards compatibility."""
    return service
