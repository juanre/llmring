"""Test configuration and fixtures for llmring"""

import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from httpx import ASGITransport, AsyncClient

from llmring.providers.anthropic_api import AnthropicProvider
from llmring.providers.google_api import GoogleProvider
from llmring.providers.ollama_api import OllamaProvider
from llmring.providers.openai_api import OpenAIProvider
from llmring.schemas import LLMResponse, Message
from llmring.service import LLMRing

# Load environment variables from .env for test runs
load_dotenv()

# -----------------------------------------------------------------------------
# llmring-server test fixtures
# -----------------------------------------------------------------------------

# Check if llmring-server is available for integration tests
try:
    import llmring_server
    from llmring_server.main import app as _server_app
    from pgdbm.fixtures.conftest import test_db_factory  # noqa: F401
    from pgdbm.migrations import AsyncMigrationManager

    LLMRING_SERVER_AVAILABLE = True
except ImportError:
    LLMRING_SERVER_AVAILABLE = False


@pytest_asyncio.fixture
async def llmring_server_client(test_db_factory) -> AsyncGenerator[AsyncClient, None]:
    """Run the llmring-server FastAPI app in-process with isolated test database.

    This fixture creates a fresh database for each test, applies migrations,
    and provides an HTTP client to interact with the server.
    """
    if not LLMRING_SERVER_AVAILABLE:
        pytest.skip("llmring-server not installed, skipping integration tests")

    # Import inside fixture to avoid import errors when server not available
    from llmring_server.main import app as server_app
    from pathlib import Path

    # Create isolated test database with schema
    db = await test_db_factory.create_db(suffix="llmring", schema="llmring_test")

    # Apply llmring-server migrations
    # We locate the migrations relative to the llmring_server module
    import llmring_server

    server_path = Path(llmring_server.__file__).parent
    migrations_path = server_path / "migrations"

    if not migrations_path.exists():
        pytest.fail(f"Migrations not found at {migrations_path}")

    migrations = AsyncMigrationManager(
        db, migrations_path=str(migrations_path), module_name="llmring_test"
    )
    await migrations.apply_pending_migrations()

    # Inject the test database into the app
    server_app.state.db = db

    # Create ASGI transport and client
    transport = ASGITransport(app=server_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def project_headers():
    """Default project header for key-scoped server routes."""
    return {"X-Project-Key": "proj_test"}


@pytest_asyncio.fixture
async def seeded_server(llmring_server_client, project_headers):
    """Seed the server with a couple of aliases and one usage log.

    - Creates aliases: summarizer → fast, cheap → fast
    - Logs one usage entry (with explicit cost to avoid live registry dependency)
    """
    c = llmring_server_client

    # Seed aliases
    for alias, model in [
        ("summarizer", "fast"),
        ("cheap", "fast"),
    ]:
        r = await c.post(
            "/api/v1/aliases/bind",
            json={"alias": alias, "model": model},
            headers=project_headers,
        )
        assert r.status_code == 200

    # Seed one usage log with explicit cost so test does not hit remote registry
    r = await c.post(
        "/api/v1/log",
        json={
            "model": "fast",
            "provider": "openai",
            "input_tokens": 100,
            "output_tokens": 20,
            "cached_input_tokens": 0,
            "cost": 0.0001,
            "alias": "summarizer",
            "profile": "default",
            "origin": "llmring-tests",
        },
        headers=project_headers,
    )
    assert r.status_code == 200

    # Yield the prepared client for tests that need it
    yield llmring_server_client


# Provider fixtures
@pytest.fixture
def openai_provider():
    """Create OpenAI provider instance, skip if no API key"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment")
    return OpenAIProvider(api_key=api_key)


@pytest.fixture
def anthropic_provider():
    """Create Anthropic provider instance, skip if no API key"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found in environment")
    return AnthropicProvider(api_key=api_key)


@pytest.fixture
def google_provider():
    """Create Google provider instance, skip if no API key"""
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GEMINI_API_KEY")
    )
    if not api_key:
        pytest.skip(
            "GOOGLE_API_KEY, GEMINI_API_KEY, or GOOGLE_GEMINI_API_KEY not found in environment"
        )
    return GoogleProvider(api_key=api_key)


@pytest.fixture
def ollama_provider():
    """Create Ollama provider instance"""
    return OllamaProvider()


# Test data fixtures
@pytest.fixture
def simple_user_message():
    """Return a simple user message for basic tests"""
    return [Message(role="user", content="Say exactly 'Test successful'")]


@pytest.fixture
def system_user_messages():
    """Return messages with system prompt"""
    return [
        Message(
            role="system",
            content="You are a helpful assistant. Always respond with exactly '4' when asked for 2+2.",
        ),
        Message(role="user", content="What is 2+2?"),
    ]


@pytest.fixture
def multi_turn_conversation():
    """Return a multi-turn conversation"""
    return [
        Message(role="user", content="My name is Alice"),
        Message(
            role="assistant",
            content="Nice to meet you, Alice! How can I help you today?",
        ),
        Message(role="user", content="What's my name?"),
    ]


@pytest.fixture
def sample_tools():
    """Return sample function definitions for tool testing"""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]


@pytest.fixture
def json_response_format():
    """Return JSON response format configuration"""
    return {
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer"],
        },
    }


@pytest.fixture
def sample_llm_response():
    """Return a sample LLM response for validation"""
    return LLMResponse(
        content="This is a test response",
        model="test-model",
        usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        finish_reason="stop",
    )


# Service fixtures
@pytest.fixture
async def llmring():
    """Create LLMRing instance"""
    service = LLMRing()
    yield service
    await service.close()


# Async test configuration
@pytest.fixture
def anyio_backend():
    """Configure async test backend"""
    return "asyncio"
