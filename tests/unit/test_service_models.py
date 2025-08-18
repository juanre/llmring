"""
Unit tests for LLM service model retrieval methods.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest
from llmring.schemas import LLMModel
from llmring.service import LLMRing


class TestLLMRingModelMethods:
    """Unit tests for LLMRing model retrieval methods."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        db = Mock()
        db.initialize = AsyncMock()
        db.apply_migrations = AsyncMock()
        db.list_models = AsyncMock()
        db.get_model = AsyncMock()
        return db

    @pytest.fixture
    def llmring_with_mock_db(self, mock_db):
        """Create LLMRing with mocked database."""
        service = LLMRing(enable_db_logging=True)
        service.db = mock_db
        return service

    @pytest.fixture
    def sample_models(self):
        """Create sample model data."""
        return [
            LLMModel(
                id=1,
                provider="openai",
                model_name="gpt-4",
                display_name="GPT-4",
                description="OpenAI's GPT-4 model",
                max_context=8192,
                max_output_tokens=4096,
                supports_vision=False,
                supports_function_calling=True,
                supports_json_mode=True,
                dollars_per_million_tokens_input=Decimal("30.00"),
                dollars_per_million_tokens_output=Decimal("60.00"),
                is_active=True,
            ),
            LLMModel(
                id=2,
                provider="anthropic",
                model_name="claude-3-opus",
                display_name="Claude 3 Opus",
                description="Anthropic's most capable model",
                max_context=200000,
                max_output_tokens=4096,
                supports_vision=True,
                supports_function_calling=True,
                supports_json_mode=False,
                dollars_per_million_tokens_input=Decimal("15.00"),
                dollars_per_million_tokens_output=Decimal("75.00"),
                is_active=True,
            ),
        ]

    @pytest.mark.skip(reason="Database functionality moving to llmring-server")
    @pytest.mark.asyncio
    async def test_get_models_from_db_success(
        self, llmring_with_mock_db, sample_models
    ):
        """Test successful model retrieval from database."""
        service = llmring_with_mock_db
        service.db.list_models.return_value = sample_models

        # Call the method
        models = await service.get_models_from_db()

        # Verify database was initialized
        assert service.db.initialize.called
        assert service.db.apply_migrations.called

        # Verify list_models was called with correct parameters
        service.db.list_models.assert_called_once_with(provider=None, active_only=True)

        # Verify results
        assert models == sample_models
        assert len(models) == 2

    @pytest.mark.asyncio
    async def test_get_models_from_db_with_provider(
        self, llmring_with_mock_db, sample_models
    ):
        """Test model retrieval filtered by provider."""
        service = llmring_with_mock_db
        openai_models = [m for m in sample_models if m.provider == "openai"]
        service.db.list_models.return_value = openai_models

        # Call the method with provider filter
        models = await service.get_models_from_db(provider="openai")

        # Verify list_models was called with provider
        service.db.list_models.assert_called_once_with(
            provider="openai", active_only=True
        )

        # Verify results
        assert len(models) == 1
        assert models[0].provider == "openai"

    @pytest.mark.asyncio
    async def test_get_models_from_db_include_inactive(
        self, llmring_with_mock_db, sample_models
    ):
        """Test model retrieval including inactive models."""
        service = llmring_with_mock_db
        service.db.list_models.return_value = sample_models

        # Call the method with active_only=False
        models = await service.get_models_from_db(active_only=False)

        # Verify list_models was called with active_only=False
        service.db.list_models.assert_called_once_with(provider=None, active_only=False)

    @pytest.mark.asyncio
    async def test_get_models_from_db_no_database(self):
        """Test model retrieval when database is not configured."""
        service = LLMRing(enable_db_logging=False)

        # Should return empty list
        models = await service.get_models_from_db()
        assert models == []

    @pytest.mark.asyncio
    async def test_get_models_from_db_error_handling(self, llmring_with_mock_db):
        """Test error handling in model retrieval."""
        service = llmring_with_mock_db
        service.db.list_models.side_effect = Exception("Database connection failed")

        # Should raise the exception
        with pytest.raises(Exception, match="Database connection failed"):
            await service.get_models_from_db()

    @pytest.mark.skip(reason="Database functionality moving to llmring-server")
    @pytest.mark.asyncio
    async def test_get_model_from_db_success(
        self, llmring_with_mock_db, sample_models
    ):
        """Test successful single model retrieval."""
        service = llmring_with_mock_db
        service.db.get_model.return_value = sample_models[0]

        # Call the method
        model = await service.get_model_from_db("openai", "gpt-4")

        # Verify database was initialized
        assert service.db.initialize.called

        # Verify get_model was called with correct parameters
        service.db.get_model.assert_called_once_with("openai", "gpt-4")

        # Verify result
        assert model == sample_models[0]
        assert model.provider == "openai"
        assert model.model_name == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_model_from_db_not_found(self, llmring_with_mock_db):
        """Test single model retrieval when model not found."""
        service = llmring_with_mock_db
        service.db.get_model.return_value = None

        # Call the method
        model = await service.get_model_from_db("openai", "non-existent")

        # Verify result
        assert model is None

    @pytest.mark.asyncio
    async def test_get_model_from_db_no_database(self):
        """Test single model retrieval when database is not configured."""
        service = LLMRing(enable_db_logging=False)

        # Should return None
        model = await service.get_model_from_db("openai", "gpt-4")
        assert model is None

    @pytest.mark.asyncio
    async def test_get_model_from_db_error_handling(self, llmring_with_mock_db):
        """Test error handling in single model retrieval."""
        service = llmring_with_mock_db
        service.db.get_model.side_effect = Exception("Database query failed")

        # Should raise the exception
        with pytest.raises(Exception, match="Database query failed"):
            await service.get_model_from_db("openai", "gpt-4")

    @pytest.mark.skip(reason="Database functionality moving to llmring-server")
    @pytest.mark.asyncio
    async def test_database_initialization_only_once(self, llmring_with_mock_db):
        """Test that database is initialized only once."""
        service = llmring_with_mock_db
        service.db.list_models.return_value = []

        # Call multiple times
        await service.get_models_from_db()
        await service.get_models_from_db()
        await service.get_model_from_db("openai", "gpt-4")

        # Database should be initialized only once
        assert service.db.initialize.call_count == 1
        assert service.db.apply_migrations.call_count == 1

    @pytest.mark.asyncio
    async def test_logging_in_model_methods(
        self, llmring_with_mock_db, sample_models, caplog
    ):
        """Test that appropriate logs are generated."""
        import logging

        service = llmring_with_mock_db
        service.db.list_models.return_value = sample_models

        with caplog.at_level(logging.INFO):
            # Test get_models_from_db logging
            await service.get_models_from_db(provider="openai")
            assert "Fetching models from database - provider: openai" in caplog.text
            assert "Successfully fetched 2 models from database" in caplog.text

        caplog.clear()

        with caplog.at_level(logging.INFO):
            # Test get_model_from_db logging
            service.db.get_model.return_value = sample_models[0]
            await service.get_model_from_db("openai", "gpt-4")
            assert (
                "Fetching model from database - provider: openai, model: gpt-4"
                in caplog.text
            )
