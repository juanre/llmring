"""Test configuration and fixtures for llmring"""

import os

import pytest
from dotenv import load_dotenv

# Load environment variables from .env for test runs
load_dotenv()

# Import async-db-utils testing fixtures
from pgdbm.fixtures.conftest import *  # Import all async-db-utils test fixtures
from llmring.db import LLMDatabase
from llmring.providers.anthropic_api import AnthropicProvider
from llmring.providers.google_api import GoogleProvider
from llmring.providers.ollama_api import OllamaProvider
from llmring.providers.openai_api import OpenAIProvider
from llmring.schemas import LLMResponse, Message
from llmring.service import LLMRing


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
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment")
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


# Database fixtures
@pytest.fixture
async def llm_test_db(test_db_factory):
    """Create test database with LLM service schema and migrations."""
    # Create database for llmring schema
    db_manager = await test_db_factory.create_db(suffix="llm", schema="llmring")

    # Apply LLM service migrations
    from pgdbm import AsyncMigrationManager

    migration_manager = AsyncMigrationManager(
        db_manager, migrations_path="src/llmring/migrations", module_name="llmring"
    )
    await migration_manager.apply_pending_migrations()

    yield {"db_manager": db_manager, "db_url": db_manager.config.get_dsn()}


@pytest.fixture
async def llm_database(llm_test_db):
    """Create LLMDatabase instance connected to test database"""
    # Use the sync wrapper which will handle async internally
    return LLMDatabase(connection_string=llm_test_db["db_url"])


@pytest.fixture
async def llmring_with_db(llm_test_db):
    """Create LLMRing instance with test database"""
    # Set environment variables for providers if available
    providers_configured = []

    service = LLMRing(
        db_connection_string=llm_test_db["db_url"],
        origin="test-suite",
        enable_db_logging=True,
    )

    yield service
    await service.close()


@pytest.fixture
async def llmring_no_db():
    """Create LLMRing instance without database logging"""
    service = LLMRing(enable_db_logging=False)
    yield service
    await service.close()


# Async test configuration
@pytest.fixture
def anyio_backend():
    """Configure async test backend"""
    return "asyncio"
