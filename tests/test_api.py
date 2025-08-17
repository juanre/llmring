"""Tests for the LLM Service API."""

from datetime import datetime
from decimal import Decimal

import pytest
from llmring.api.service import LLMRingAPI
from llmring.api.types import (
    CostBreakdown,
    ModelInfo,
    ModelRequirements,
    ServiceHealth,
)
from llmring.schemas import LLMModel


@pytest.fixture
async def api_service(llm_database):
    """Create an API service instance with test database."""
    await llm_database.initialize()
    api = LLMRingAPI(llm_database)
    return api


@pytest.fixture
async def sample_models(llm_database):
    """Add sample models to the database."""
    await llm_database.initialize()

    # Clear any existing models
    await llm_database.db.execute("DELETE FROM llmring.llm_models")

    models = [
        {
            "provider": "openai",
            "model_name": "gpt-4o",
            "display_name": "GPT-4o",
            "description": "Latest GPT-4 model",
            "max_context": 128000,
            "max_output_tokens": 4096,
            "supports_vision": True,
            "supports_function_calling": True,
            "supports_json_mode": True,
            "dollars_per_million_tokens_input": Decimal("5.00"),
            "dollars_per_million_tokens_output": Decimal("15.00"),
            "is_active": True,
        },
        {
            "provider": "openai",
            "model_name": "gpt-3.5-turbo",
            "display_name": "GPT-3.5 Turbo",
            "description": "Fast and affordable",
            "max_context": 16384,
            "max_output_tokens": 4096,
            "supports_function_calling": True,
            "supports_json_mode": True,
            "dollars_per_million_tokens_input": Decimal("0.50"),
            "dollars_per_million_tokens_output": Decimal("1.50"),
            "is_active": True,
        },
        {
            "provider": "anthropic",
            "model_name": "claude-3-opus-20240229",
            "display_name": "Claude 3 Opus",
            "description": "Most capable Claude model",
            "max_context": 200000,
            "max_output_tokens": 4096,
            "supports_vision": True,
            "supports_function_calling": True,
            "dollars_per_million_tokens_input": Decimal("15.00"),
            "dollars_per_million_tokens_output": Decimal("75.00"),
            "is_active": True,
        },
        {
            "provider": "anthropic",
            "model_name": "claude-3-haiku-20240307",
            "display_name": "Claude 3 Haiku",
            "description": "Fast and affordable",
            "max_context": 200000,
            "max_output_tokens": 4096,
            "supports_vision": True,
            "supports_function_calling": True,
            "dollars_per_million_tokens_input": Decimal("0.25"),
            "dollars_per_million_tokens_output": Decimal("1.25"),
            "is_active": False,  # Inactive for testing
        },
        {
            "provider": "ollama",
            "model_name": "llama3.2",
            "display_name": "Llama 3.2",
            "description": "Local model",
            "max_context": 8192,
            "max_output_tokens": 2048,
            "supports_function_calling": True,
            "is_active": True,
        },
    ]

    # Store created models for reference
    created_models = []

    for model_data in models:
        # Remove is_active - we'll handle it separately
        is_active = model_data.pop("is_active", True)

        # Create model with schema fields
        model = LLMModel(
            provider=model_data["provider"],
            model_name=model_data["model_name"],
            display_name=model_data.get("display_name"),
            description=model_data.get("description"),
            max_context=model_data.get("max_context"),
            max_output_tokens=model_data.get("max_output_tokens"),
            supports_vision=model_data.get("supports_vision", False),
            supports_function_calling=model_data.get(
                "supports_function_calling", False
            ),
            supports_json_mode=model_data.get("supports_json_mode", False),
            dollars_per_million_tokens_input=model_data.get(
                "dollars_per_million_tokens_input"
            ),
            dollars_per_million_tokens_output=model_data.get(
                "dollars_per_million_tokens_output"
            ),
            inactive_from=None if is_active else datetime.now(),
        )

        # Add model will handle conversion to inactive_from
        await llm_database.add_model(model)
        created_models.append(model)

    return models


class TestModelDiscovery:
    """Test model discovery and retrieval methods."""

    async def test_list_models(self, api_service, sample_models):
        """Test listing models with various filters."""
        # List all active models
        models = await api_service.list_models()
        assert len(models) == 4  # 5 total, 1 inactive

        # List all models including inactive
        all_models = await api_service.list_models(active_only=False)
        assert len(all_models) == 5

        # List by provider
        openai_models = await api_service.list_models(provider="openai")
        assert len(openai_models) == 2
        assert all(m.provider == "openai" for m in openai_models)

        # Test sorting
        sorted_models = await api_service.list_models(sort_by="cost", sort_order="asc")
        costs = [
            m.cost_per_million_input_tokens
            for m in sorted_models
            if m.cost_per_million_input_tokens
        ]
        assert costs == sorted(costs)

    async def test_get_model(self, api_service, sample_models):
        """Test getting a specific model."""
        # Get existing model
        model = await api_service.get_model("openai", "gpt-4o")
        assert model is not None
        assert model.provider == "openai"
        assert model.model_name == "gpt-4o"
        assert model.display_name == "GPT-4o"
        assert model.supports_vision is True
        assert model.has_pricing is True

        # Get non-existent model
        model = await api_service.get_model("openai", "gpt-5")
        assert model is None

    async def test_get_providers(self, api_service, sample_models):
        """Test getting provider information."""
        providers = await api_service.get_providers()
        assert len(providers) == 3  # openai, anthropic, ollama

        # Check OpenAI provider
        openai = next(p for p in providers if p.provider == "openai")
        assert openai.total_models == 2
        assert openai.active_models == 2
        assert "vision" in openai.features
        assert "functions" in openai.features

    async def test_batch_get_models(self, api_service, sample_models):
        """Test batch model retrieval."""
        requests = [
            ("openai", "gpt-4o"),
            ("anthropic", "claude-3-opus-20240229"),
            ("openai", "gpt-5"),  # Non-existent
        ]

        results = await api_service.batch_get_models(requests)
        assert len(results) == 3
        assert results[("openai", "gpt-4o")] is not None
        assert results[("anthropic", "claude-3-opus-20240229")] is not None
        assert results[("openai", "gpt-5")] is None


class TestModelStatistics:
    """Test model statistics and summaries."""

    async def test_get_model_statistics(self, api_service, sample_models):
        """Test comprehensive model statistics."""
        stats = await api_service.get_model_statistics()

        assert stats.total_models == 5
        assert stats.active_models == 4
        assert stats.inactive_models == 1
        assert stats.models_with_pricing == 4
        assert stats.models_without_pricing == 1

        # Check provider stats
        assert "openai" in stats.providers
        openai_stats = stats.providers["openai"]
        assert openai_stats.total_models == 2
        assert openai_stats.active_models == 2
        assert openai_stats.vision_models == 1
        assert openai_stats.function_calling_models == 2
        assert openai_stats.min_input_cost == 0.5
        assert openai_stats.max_input_cost == 5.0

    async def test_get_cost_statistics(self, api_service, sample_models):
        """Test cost statistics calculation."""
        costs = await api_service.get_cost_statistics()

        assert costs["models_with_pricing"] == 3  # Only active models
        assert costs["min_input_cost"] == 0.5  # gpt-3.5-turbo (haiku is inactive)
        assert costs["max_input_cost"] == 15.0  # claude-3-opus
        assert costs["median_input_cost"] > 0
        assert "percentiles" in costs


class TestCostCalculations:
    """Test cost calculation methods."""

    async def test_calculate_cost(self, api_service, sample_models):
        """Test cost calculation for specific usage."""
        # Test with breakdown
        cost = await api_service.calculate_cost(
            "openai",
            "gpt-4o",
            input_tokens=10000,
            output_tokens=2000,
            include_breakdown=True,
        )

        assert isinstance(cost, CostBreakdown)
        assert cost.input_tokens == 10000
        assert cost.output_tokens == 2000
        assert cost.input_cost == 0.05  # 10K * $5/M
        assert cost.output_cost == 0.03  # 2K * $15/M
        assert cost.total_cost == 0.08

        # Test without breakdown
        total_cost = await api_service.calculate_cost(
            "openai",
            "gpt-4o",
            input_tokens=10000,
            output_tokens=2000,
            include_breakdown=False,
        )
        assert total_cost == 0.08

        # Test with model without pricing
        cost = await api_service.calculate_cost(
            "ollama", "llama3.2", input_tokens=1000, output_tokens=500
        )
        assert cost is None

    async def test_compare_model_costs(self, api_service, sample_models):
        """Test cost comparison across models."""
        comparisons = await api_service.compare_model_costs(
            input_tokens=10000, output_tokens=2000
        )

        assert len(comparisons) > 0
        # Should be sorted by cost
        costs = [c.total_cost for c in comparisons]
        assert costs == sorted(costs)

        # Cheapest should be gpt-3.5-turbo (haiku is inactive)
        assert comparisons[0].model_name == "gpt-3.5-turbo"


class TestModelFiltering:
    """Test model filtering and search methods."""

    async def test_find_models_by_features(self, api_service, sample_models):
        """Test finding models by features."""
        # Find vision models
        vision_models = await api_service.find_models_by_features(vision=True)
        assert len(vision_models) == 2  # gpt-4o and claude-3-opus (haiku is inactive)
        assert all(m.supports_vision for m in vision_models)

        # Find models with function calling but not vision
        func_only = await api_service.find_models_by_features(
            vision=False, function_calling=True
        )
        assert len(func_only) == 2  # gpt-3.5-turbo and llama3.2

        # Find OpenAI models with JSON mode
        openai_json = await api_service.find_models_by_features(
            json_mode=True, providers=["openai"]
        )
        assert len(openai_json) == 2
        assert all(m.provider == "openai" for m in openai_json)

    async def test_find_models_by_cost_range(self, api_service, sample_models):
        """Test finding models by cost range."""
        # Find budget models
        budget_models = await api_service.find_models_by_cost_range(
            max_input_cost_per_million=5.0, max_output_cost_per_million=15.0
        )
        assert len(budget_models) == 2  # gpt-3.5-turbo and gpt-4o

        # Find premium models
        premium_models = await api_service.find_models_by_cost_range(
            min_input_cost_per_million=10.0
        )
        assert len(premium_models) == 1  # claude-3-opus

    async def test_find_models_by_context_size(self, api_service, sample_models):
        """Test finding models by context size."""
        # Find large context models
        large_context = await api_service.find_models_by_context_size(
            min_context=100000
        )
        assert len(large_context) == 2  # gpt-4o and claude-3-opus

        # Find models with specific output size
        large_output = await api_service.find_models_by_context_size(min_output=4000)
        assert len(large_output) == 3  # All except llama3.2

    async def test_search_models(self, api_service, sample_models):
        """Test text search for models."""
        # Search for "gpt"
        results = await api_service.search_models("gpt")
        assert len(results) == 2
        assert all("gpt" in m.model_name.lower() for m in results)

        # Search for "fast"
        results = await api_service.search_models("fast")
        assert len(results) == 1  # gpt-3.5-turbo has "fast" in description

        # Fuzzy search
        results = await api_service.search_models("claude opus", fuzzy=True)
        assert len(results) >= 1

    async def test_find_compatible_models(self, api_service, sample_models):
        """Test finding models meeting requirements."""
        requirements = ModelRequirements(
            min_context_size=50000,
            max_input_cost_per_million=10.0,
            requires_vision=True,
            requires_function_calling=True,
            active_only=True,
        )

        compatible = await api_service.find_compatible_models(requirements)
        assert len(compatible) == 1  # Only gpt-4o meets all requirements
        assert compatible[0].model_name == "gpt-4o"


class TestModelManagement:
    """Test model management methods."""

    async def test_activate_deactivate_model(self, api_service, sample_models):
        """Test activating and deactivating models."""
        # Activate inactive model
        success = await api_service.activate_model(
            "anthropic", "claude-3-haiku-20240307"
        )
        assert success is True

        # Verify it's active
        model = await api_service.get_model("anthropic", "claude-3-haiku-20240307")
        assert model.is_active is True

        # Deactivate it
        success = await api_service.deactivate_model(
            "anthropic", "claude-3-haiku-20240307"
        )
        assert success is True

        # Verify it's inactive
        model = await api_service.get_model("anthropic", "claude-3-haiku-20240307")
        assert model.is_active is False

    async def test_bulk_update_model_status(self, api_service, sample_models):
        """Test bulk status updates."""
        updates = [
            ("anthropic", "claude-3-haiku-20240307", True),  # Activate
            ("openai", "gpt-3.5-turbo", False),  # Deactivate
        ]

        results = await api_service.bulk_update_model_status(updates)
        assert results[("anthropic", "claude-3-haiku-20240307")] is True
        assert results[("openai", "gpt-3.5-turbo")] is True

        # Verify changes
        haiku = await api_service.get_model("anthropic", "claude-3-haiku-20240307")
        assert haiku.is_active is True

        gpt35 = await api_service.get_model("openai", "gpt-3.5-turbo")
        assert gpt35.is_active is False

    async def test_activate_all_models(self, api_service, sample_models):
        """Test activating all models."""
        # First deactivate some
        await api_service.deactivate_model("openai", "gpt-4o")
        await api_service.deactivate_model("anthropic", "claude-3-opus-20240229")

        # Activate all OpenAI models
        count = await api_service.activate_all_models("openai")
        assert count == 1  # Only gpt-4o was inactive

        # Activate all models
        count = await api_service.activate_all_models()
        assert count == 2  # claude-3-opus and claude-3-haiku (was initially inactive)


class TestModelValidation:
    """Test model validation and recommendations."""

    async def test_validate_model_request(self, api_service, sample_models):
        """Test model request validation."""
        # Valid model without requirements
        result = await api_service.validate_model_request("openai", "gpt-4o")
        assert result.valid is True
        assert result.model_exists is True
        assert result.model_active is True

        # Non-existent model
        result = await api_service.validate_model_request("openai", "gpt-5")
        assert result.valid is False
        assert result.model_exists is False
        assert len(result.issues) > 0

        # Model with requirements
        requirements = ModelRequirements(min_context_size=150000, requires_vision=True)
        result = await api_service.validate_model_request(
            "openai", "gpt-4o", requirements
        )
        assert result.valid is False  # Context too small
        assert result.meets_requirements is False
        assert "Context size" in result.issues[0]

    async def test_suggest_alternative_models(self, api_service, sample_models):
        """Test model alternative suggestions."""
        # Alternatives for expensive model
        alternatives = await api_service.suggest_alternative_models(
            "anthropic", "claude-3-opus-20240229"
        )
        assert len(alternatives) > 0
        # Should prefer models with similar features
        assert any(m.supports_vision for m in alternatives[:2])

    async def test_get_model_recommendations(self, api_service, sample_models):
        """Test model recommendations for use cases."""
        # Code generation recommendations
        recommendations = await api_service.get_model_recommendations(
            use_case="code_generation", budget_per_million_tokens=10.0
        )

        assert len(recommendations) > 0
        # Should return tuples of (model, score)
        assert all(isinstance(r[1], float) for r in recommendations)
        # Should prefer models with function calling
        top_model = recommendations[0][0]
        assert top_model.supports_function_calling is True


class TestServiceHealth:
    """Test service health and monitoring."""

    async def test_get_service_health(self, api_service, sample_models):
        """Test service health check."""
        health = await api_service.get_service_health()

        assert isinstance(health, ServiceHealth)
        assert health.status == "healthy"
        assert health.database_connected is True
        assert health.models_loaded == 5
        assert len(health.issues) == 0

    async def test_verify_model_data_integrity(self, api_service, sample_models):
        """Test data integrity verification."""
        result = await api_service.verify_model_data_integrity()

        assert "total_models" in result
        assert result["total_models"] == 5
        assert "issues" in result
        # Our test data should have no integrity issues
        assert len(result["issues"]) == 0


class TestUtilityMethods:
    """Test utility methods."""

    async def test_get_model_families(self, api_service, sample_models):
        """Test model family grouping."""
        families = await api_service.get_model_families()

        assert "gpt" in families
        assert len(families["gpt"]) == 2
        assert "claude" in families
        assert len(families["claude"]) == 2

    async def test_normalize_model_name(self, api_service, sample_models):
        """Test model name normalization."""
        # Exact match
        name = await api_service.normalize_model_name("openai", "gpt-4o")
        assert name == "gpt-4o"

        # Common alias
        name = await api_service.normalize_model_name("anthropic", "claude-3-opus")
        assert name == "claude-3-opus-20240229"

        # Partial match
        name = await api_service.normalize_model_name("openai", "3.5")
        assert name == "gpt-3.5-turbo"

        # No match
        name = await api_service.normalize_model_name("openai", "gpt-5")
        assert name is None

    async def test_estimate_tokens(self, api_service, sample_models):
        """Test token estimation."""
        text = "Hello, world! This is a test message."

        # OpenAI estimation
        tokens = await api_service.estimate_tokens(text, "openai", "gpt-4o")
        assert tokens > 0
        assert tokens < len(text)  # Should be less than character count

        # Anthropic estimation (slightly different)
        tokens_anthropic = await api_service.estimate_tokens(
            text, "anthropic", "claude-3-opus"
        )
        assert tokens_anthropic > 0
        # Anthropic typically uses more tokens
        assert tokens_anthropic >= tokens


class TestModelInfo:
    """Test ModelInfo type functionality."""

    def test_model_info_cost_calculations(self):
        """Test ModelInfo cost calculation methods."""
        model = ModelInfo(
            provider="openai",
            model_name="gpt-4o",
            display_name="GPT-4o",
            is_active=True,
            last_updated=datetime.now(),
            added_date=datetime.now(),
            cost_per_million_input_tokens=Decimal("5.00"),
            cost_per_million_output_tokens=Decimal("15.00"),
        )

        # Test cost per unit
        input_1k, output_1k = model.get_cost_per_unit("1k")
        assert input_1k == 0.005
        assert output_1k == 0.015

        input_token, output_token = model.get_cost_per_unit("token")
        assert input_token == 0.000005
        assert output_token == 0.000015

        # Test cost calculation
        cost = model.calculate_cost(input_tokens=10000, output_tokens=2000)
        assert cost == 0.08  # (10K * $5/M) + (2K * $15/M)

        # Test format string
        display = model.format_cost_string("1k")
        assert display == "$0.005/$0.015 per 1k tokens"

        display_1m = model.format_cost_string("1m")
        assert display_1m == "$5.000/$15.000 per 1M tokens"
