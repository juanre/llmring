"""
Test configuration and fixtures for mcp-client.

This module provides test fixtures for mcp-client tests using HTTP-based architecture.
"""

import pytest
from llmring.schemas import LLMRequest, LLMResponse, Message
from llmring.service import LLMRing
from llmring.mcp.client.stateless_engine import StatelessChatEngine


@pytest.fixture
def llm_service():
    """Create an LLMRing instance for testing."""
    import os
    test_lockfile = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'llmring.lock.json'
    )
    return LLMRing(origin="test", lockfile_path=test_lockfile)


@pytest.fixture
async def stateless_engine(llm_service):
    """Create a StatelessChatEngine for testing."""
    return StatelessChatEngine(llmring=llm_service)


@pytest.fixture
def simple_messages():
    """Simple test messages."""
    return [Message(role="user", content="Hello, how are you?")]


@pytest.fixture
def test_auth_context():
    """Test authentication context."""
    return {
        "user_id": "test-user-123",
        "user_info": {"type": "test", "name": "Test User"},
    }


@pytest.fixture
def sample_llm_request():
    """Sample LLM request for testing."""
    return LLMRequest(
        messages=[Message(role="user", content="Test message")],
        model="claude-3-sonnet-20240229",
        temperature=0.7,
        max_tokens=100,
    )


@pytest.fixture
def sample_llm_response():
    """Sample LLM response for testing."""
    return LLMResponse(
        content="Test response",
        model="claude-3-sonnet-20240229",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        finish_reason="stop",
    )
