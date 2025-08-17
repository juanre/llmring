"""
Integration tests for LLM service database model operations.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal

import pytest
from llmring.schemas import LLMModel
from llmring.service import LLMRing

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.asyncio
class TestLLMRingDatabaseIntegration:
    """Integration tests for LLMRing database operations."""

    @pytest.fixture
    async def llmring(self, llm_test_db):
        """Create LLMRing with database for testing."""
        service = LLMRing(
            db_connection_string=llm_test_db["db_url"],
            origin="test-llmring",
            enable_db_logging=True,
        )
        yield service
        # Cleanup
        if service.db and service._db_initialized:
            await service.close()

    @pytest.fixture
    async def sample_models(self, llmring):
        """Create sample models for testing."""
        # Ensure database is initialized
        await llmring._ensure_db_initialized()

        # Add test models
        test_models = [
            LLMModel(
                provider="openai",
                model_name="test-gpt-4",
                display_name="Test GPT-4",
                description="Test model for GPT-4",
                max_context=8192,
                max_output_tokens=4096,
                supports_vision=False,
                supports_function_calling=True,
                supports_json_mode=True,
                dollars_per_million_tokens_input=Decimal("30.00"),
                dollars_per_million_tokens_output=Decimal("60.00"),
            ),
            LLMModel(
                provider="anthropic",
                model_name="test-claude-3",
                display_name="Test Claude 3",
                description="Test model for Claude",
                max_context=200000,
                max_output_tokens=4096,
                supports_vision=True,
                supports_function_calling=True,
                supports_json_mode=False,
                dollars_per_million_tokens_input=Decimal("10.00"),
                dollars_per_million_tokens_output=Decimal("20.00"),
            ),
            LLMModel(
                provider="openai",
                model_name="test-gpt-3.5-inactive",
                display_name="Test GPT-3.5 (Inactive)",
                description="Inactive test model",
                max_context=4096,
                max_output_tokens=2048,
                supports_vision=False,
                supports_function_calling=False,
                supports_json_mode=False,
                dollars_per_million_tokens_input=Decimal("1.00"),
                dollars_per_million_tokens_output=Decimal("2.00"),
                inactive_from=datetime.now(),
            ),
        ]

        # Add models to database
        for model in test_models:
            await llmring.db.add_model(model)

        return test_models

    async def test_get_models_from_db_all(self, llmring, sample_models):
        """Test fetching all models from database."""
        # Test fetching all active models
        models = await llmring.get_models_from_db()

        assert isinstance(models, list)
        assert len(models) >= 2  # At least our active test models

        # Check that inactive models are filtered out by default
        model_names = [m.model_name for m in models]
        assert "test-gpt-4" in model_names
        assert "test-claude-3" in model_names
        assert "test-gpt-3.5-inactive" not in model_names

    async def test_get_models_from_db_with_inactive(self, llmring, sample_models):
        """Test fetching models including inactive ones."""
        # Test fetching all models including inactive
        models = await llmring.get_models_from_db(active_only=False)

        assert isinstance(models, list)
        assert len(models) >= 3  # Including inactive model

        model_names = [m.model_name for m in models]
        assert "test-gpt-4" in model_names
        assert "test-claude-3" in model_names
        assert "test-gpt-3.5-inactive" in model_names

    async def test_get_models_from_db_by_provider(self, llmring, sample_models):
        """Test fetching models filtered by provider."""
        # Test fetching OpenAI models only
        openai_models = await llmring.get_models_from_db(provider="openai")

        assert isinstance(openai_models, list)
        assert all(m.provider == "openai" for m in openai_models)
        assert any(m.model_name == "test-gpt-4" for m in openai_models)

        # Test fetching Anthropic models only
        anthropic_models = await llmring.get_models_from_db(provider="anthropic")

        assert isinstance(anthropic_models, list)
        assert all(m.provider == "anthropic" for m in anthropic_models)
        assert any(m.model_name == "test-claude-3" for m in anthropic_models)

    async def test_get_model_from_db(self, llmring, sample_models):
        """Test fetching a specific model from database."""
        # Test fetching existing model
        model = await llmring.get_model_from_db("openai", "test-gpt-4")

        assert model is not None
        assert model.provider == "openai"
        assert model.model_name == "test-gpt-4"
        assert model.display_name == "Test GPT-4"
        assert model.supports_function_calling is True
        assert model.is_active is True

    async def test_get_model_from_db_not_found(self, llmring, sample_models):
        """Test fetching a non-existent model."""
        # Test fetching non-existent model
        model = await llmring.get_model_from_db("openai", "non-existent-model")

        assert model is None

    async def test_get_model_from_db_inactive(self, llmring, sample_models):
        """Test that inactive models are not returned."""
        # The get_model method in db.py filters by is_active=TRUE
        model = await llmring.get_model_from_db("openai", "test-gpt-3.5-inactive")

        assert model is None  # Should not return inactive models

    async def test_database_not_configured(self):
        """Test behavior when database is not configured."""
        # Create service without database
        service = LLMRing(enable_db_logging=False)

        # Test get_models_from_db
        models = await service.get_models_from_db()
        assert models == []

        # Test get_model_from_db
        model = await service.get_model_from_db("openai", "gpt-4")
        assert model is None

    async def test_database_initialization_error_handling(self, llm_test_db, caplog):
        """Test error handling during database initialization."""
        # Create service with invalid migrations path
        service = LLMRing(
            db_connection_string=llm_test_db["db_url"] + "_invalid",
            origin="test-error",
            enable_db_logging=True,
        )

        with caplog.at_level(logging.ERROR):
            # Try to fetch models - should handle initialization error gracefully
            try:
                await service.get_models_from_db()
            except Exception:
                # Expected to raise since we re-raise in the method
                pass

            # Check that error was logged
            assert (
                "Failed to initialize database" in caplog.text
                or "Failed to fetch models" in caplog.text
            )

    async def test_concurrent_model_fetches(self, llmring, sample_models):
        """Test concurrent database operations."""
        # Create multiple concurrent requests
        agents = [
            llmring.get_models_from_db(),
            llmring.get_models_from_db(provider="openai"),
            llmring.get_model_from_db("anthropic", "test-claude-3"),
            llmring.get_models_from_db(active_only=False),
        ]

        # Execute concurrently
        results = await asyncio.gather(*agents)

        # Verify results
        assert len(results) == 4
        assert isinstance(results[0], list)  # All models
        assert isinstance(results[1], list)  # OpenAI models
        assert results[2] is not None  # Specific model
        assert isinstance(results[3], list)  # All models including inactive
