"""Tests for database functionality."""

from decimal import Decimal
from uuid import UUID

from llmring.schemas import LLMModel


async def test_database_initialization(llm_database):
    """Test database initialization and migration."""
    # Test that database is initialized
    assert llm_database is not None

    # Test that tables exist by querying them
    models = await llm_database.list_models()
    assert isinstance(models, list)

    # Verify we can query the tables - they should exist after migrations
    # The migrations insert default models, so we should have some
    assert len(models) > 0


async def test_model_operations(llm_database):
    """Test model CRUD operations."""
    # Test adding a model
    test_model = LLMModel(
        provider="anthropic",
        model_name="test-model",
        display_name="Test Model",
        description="A test model",
        max_context=100000,
        max_output_tokens=4096,
        supports_vision=True,
        supports_function_calling=True,
        dollars_per_million_tokens_input=Decimal("1.00"),
        dollars_per_million_tokens_output=Decimal("2.00"),
    )

    model_id = await llm_database.add_model(test_model)
    assert isinstance(model_id, int)

    # Test retrieving the model
    retrieved_model = await llm_database.get_model("anthropic", "test-model")
    assert retrieved_model is not None
    assert retrieved_model.provider == "anthropic"
    assert retrieved_model.model_name == "test-model"
    assert retrieved_model.display_name == "Test Model"
    assert retrieved_model.max_context == 100000
    assert retrieved_model.dollars_per_million_tokens_input == Decimal("1.00")

    # Test listing models
    models = await llm_database.list_models()
    assert len(models) > 0
    test_models = [m for m in models if m.model_name == "test-model"]
    assert len(test_models) == 1

    # Test filtering by provider
    anthropic_models = await llm_database.list_models(provider="anthropic")
    assert len(anthropic_models) > 0
    test_model_found = any(m.model_name == "test-model" for m in anthropic_models)
    assert test_model_found


async def test_api_call_recording(llm_database):
    """Test API call recording."""
    # Record an API call
    call_id = await llm_database.record_api_call(
        origin="test-app",
        id_at_origin="test-user",
        provider="anthropic",
        model_name="claude-3-sonnet-20240229",
        prompt_tokens=100,
        completion_tokens=50,
        response_time_ms=1500,
        temperature=0.7,
        max_tokens=1000,
        system_prompt="You are a helpful assistant",
        tools_used=["search", "calculator"],
        status="success",
    )

    assert isinstance(call_id, UUID)

    # Retrieve the call record
    call_record = await llm_database.get_call_record(call_id)
    assert call_record is not None
    assert call_record.origin == "test-app"
    assert call_record.id_at_origin == "test-user"
    assert call_record.provider == "anthropic"
    assert call_record.model_name == "claude-3-sonnet-20240229"
    assert call_record.prompt_tokens == 100
    assert call_record.completion_tokens == 50
    assert call_record.total_tokens == 150
    assert call_record.response_time_ms == 1500
    assert call_record.temperature == 0.7
    assert call_record.status == "success"
    assert call_record.estimated_cost > 0  # Should be calculated from model costs


async def test_api_call_with_error(llm_database):
    """Test API call recording with error."""
    call_id = await llm_database.record_api_call(
        origin="test-app",
        id_at_origin="test-user",
        provider="openai",
        model_name="gpt-4",
        prompt_tokens=50,
        completion_tokens=0,  # No completion due to error
        response_time_ms=500,
        status="error",
        error_type="RateLimitError",
        error_message="API rate limit exceeded",
    )

    call_record = await llm_database.get_call_record(call_id)
    assert call_record.status == "error"
    assert call_record.error_type == "RateLimitError"
    assert call_record.error_message == "API rate limit exceeded"
    assert call_record.completion_tokens == 0


async def test_usage_stats(llm_database):
    """Test usage statistics calculation."""
    # Record multiple API calls
    for i in range(5):
        await llm_database.record_api_call(
            origin="test-app",
            id_at_origin="test-user",
            provider="anthropic",
            model_name="claude-3-sonnet-20240229",
            prompt_tokens=100 + i * 10,
            completion_tokens=50 + i * 5,
            response_time_ms=1000 + i * 100,
            status="success",
        )

    # Record one error call
    await llm_database.record_api_call(
        origin="test-app",
        id_at_origin="test-user",
        provider="anthropic",
        model_name="claude-3-sonnet-20240229",
        prompt_tokens=100,
        completion_tokens=0,
        response_time_ms=100,
        status="error",
        error_type="APIError",
    )

    # Get usage stats
    stats = await llm_database.get_usage_stats("test-app", "test-user", days=1)
    assert stats is not None
    assert stats.total_calls == 6
    # Calculate expected total:
    # 5 success calls: (100+10*i + 50+5*i) for i in 0,1,2,3,4 = 150+165+180+195+210 = 900
    # 1 error call: 100 + 0 = 100
    # Total: 1000
    assert stats.total_tokens == 1000
    assert stats.most_used_model == "claude-3-sonnet-20240229"
    assert 0 < stats.success_rate < 1  # Should be 5/6
    assert stats.avg_response_time_ms > 1000


async def test_recent_calls_listing(llm_database):
    """Test listing recent API calls."""
    # Record some calls for different users
    await llm_database.record_api_call(
        origin="test-app",
        id_at_origin="user1",
        provider="anthropic",
        model_name="claude-3-sonnet-20240229",
        prompt_tokens=100,
        completion_tokens=50,
        response_time_ms=200,
    )

    await llm_database.record_api_call(
        origin="test-app",
        id_at_origin="user2",
        provider="openai",
        model_name="gpt-4",
        prompt_tokens=150,
        completion_tokens=75,
        response_time_ms=300,
    )

    # List all recent calls for the origin
    all_calls = await llm_database.list_recent_calls(origin="test-app")
    assert len(all_calls) >= 2

    # List calls for specific user
    user1_calls = await llm_database.list_recent_calls(
        origin="test-app", id_at_origin="user1"
    )
    user1_calls_filtered = [c for c in user1_calls if c.id_at_origin == "user1"]
    assert len(user1_calls_filtered) >= 1

    # Check call details
    call = user1_calls_filtered[0]
    assert call.origin == "test-app"
    assert call.id_at_origin == "user1"
    assert call.provider == "anthropic"


async def test_service_with_database(llmring_with_db, simple_user_message):
    """Test LLMRing with database logging enabled."""
    service = llmring_with_db

    # Check that database is initialized
    assert service.db is not None
    assert service.enable_db_logging is True
    assert service.origin == "test-suite"

    # Test getting models from database
    models = await service.get_models_from_db(provider="anthropic")
    assert len(models) > 0
    claude_models = [m for m in models if "claude" in m.model_name.lower()]
    assert len(claude_models) > 0

    # Test usage stats (should be empty initially)
    stats = await service.get_usage_stats("test-user")
    assert stats is None  # No calls yet

    # Test recent calls (should be empty initially)
    calls = await service.list_recent_calls("test-user")
    assert len(calls) == 0


def test_service_without_database(llmring_no_db):
    """Test LLMRing without database logging."""
    service = llmring_no_db

    # Check that database is not initialized
    assert service.db is None
    assert service.enable_db_logging is False

    # Test that database methods return empty/None
    # These methods should be async in the service too
    # For now, let's just check the service state
    assert service.db is None


async def test_default_models_inserted(llm_database):
    """Test that default models are inserted during migration."""
    # Check that default models exist
    anthropic_models = await llm_database.list_models(provider="anthropic")
    assert len(anthropic_models) > 0

    openai_models = await llm_database.list_models(provider="openai")
    assert len(openai_models) > 0

    google_models = await llm_database.list_models(provider="google")
    assert len(google_models) > 0

    ollama_models = await llm_database.list_models(provider="ollama")
    assert len(ollama_models) > 0

    # Check specific models
    claude_sonnet = await llm_database.get_model(
        "anthropic", "claude-3-sonnet-20240229"
    )
    assert claude_sonnet is not None
    assert claude_sonnet.display_name == "Claude 3 Sonnet"
    assert claude_sonnet.max_context == 200000
    assert claude_sonnet.dollars_per_million_tokens_input > 0

    gpt4 = await llm_database.get_model("openai", "gpt-4o")
    assert gpt4 is not None
    assert gpt4.supports_vision is True
    assert gpt4.supports_function_calling is True
