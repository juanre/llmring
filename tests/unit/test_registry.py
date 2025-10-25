"""
Unit tests for registry client and cost calculation.

Tests registry fetching, caching, validation, and cost calculations.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmring.registry import RegistryClient, RegistryModel
from llmring.schemas import LLMRequest, Message
from llmring.service import LLMRing


@pytest.mark.asyncio
class TestRegistryClient:
    """Test the registry client."""

    @pytest.fixture
    def test_registry_url(self):
        """Get test registry URL."""
        return f"file://{Path(__file__).parent.parent}/resources/registry"

    @pytest.fixture
    def registry_client(self, test_registry_url, tmp_path):
        """Create registry client with test data."""
        return RegistryClient(registry_url=test_registry_url, cache_dir=tmp_path / "cache")

    async def test_fetch_models_from_test_registry(self, registry_client):
        """Test fetching models from test registry files."""
        # We need to mock httpx to read from local files
        test_dir = Path(__file__).parent.parent / "resources" / "registry"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock the OpenAI models response
            openai_data = json.loads((test_dir / "openai" / "models.json").read_text())
            mock_response = MagicMock()
            mock_response.json.return_value = openai_data
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response

            # Fetch models
            models = await registry_client.fetch_current_models("openai")

            # Should have models from test registry
            assert len(models) > 0

            # Check all models have required fields
            for model in models:
                assert model.model_name
                assert model.provider == "openai"
                assert model.display_name

                # If model has pricing, verify it's valid
                if model.dollars_per_million_tokens_input:
                    assert model.dollars_per_million_tokens_input > 0
                if model.dollars_per_million_tokens_output:
                    assert model.dollars_per_million_tokens_output > 0
                if model.max_input_tokens:
                    assert model.max_input_tokens > 0

    async def test_validate_model(self, registry_client):
        """Test model validation against registry."""
        test_dir = Path(__file__).parent.parent / "resources" / "registry"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock the response
            openai_data = json.loads((test_dir / "openai" / "models.json").read_text())
            mock_response = MagicMock()
            mock_response.json.return_value = openai_data
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response

            # Get models from test registry to validate
            test_models = list(openai_data["models"].keys())
            if test_models:
                # Take first model name (strip provider prefix)
                first_model = test_models[0].replace("openai:", "")
                assert await registry_client.validate_model("openai", first_model) is True

            # Validate non-existing model
            assert await registry_client.validate_model("openai", "non-existent-model-xyz") is False

    async def test_fetch_version_from_file_scheme(self, tmp_path):
        """Ensure file:// registry versions return structured data."""
        registry_root = tmp_path / "registry"
        version_dir = registry_root / "openai" / "v" / "2"
        version_dir.mkdir(parents=True, exist_ok=True)

        version_payload = {
            "provider": "openai",
            "version": 2,
            "updated_at": "2025-09-01T00:00:00Z",
            "models": [
                {
                    "provider": "openai",
                    "model_name": "gpt-test",
                    "display_name": "GPT Test",
                    "description": "Test model",
                    "max_input_tokens": 1000,
                    "max_output_tokens": 1000,
                    "dollars_per_million_tokens_input": 0.1,
                    "dollars_per_million_tokens_output": 0.2,
                    "supports_vision": False,
                    "supports_function_calling": True,
                    "supports_json_mode": True,
                    "supports_parallel_tool_calls": False,
                    "is_active": True,
                }
            ],
        }

        (version_dir / "models.json").write_text(json.dumps(version_payload))

        registry = RegistryClient(
            registry_url=f"file://{registry_root}",
            cache_dir=tmp_path / "cache",
        )

        version_info = await registry.fetch_version("openai", 2)

        assert version_info.version == 2
        assert version_info.provider == "openai"
        assert len(version_info.models) == 1
        assert version_info.models[0].model_name == "gpt-test"

    async def test_registry_model_extended_pricing_fields(self):
        """Ensure RegistryModel captures extended pricing metadata."""
        model = RegistryModel(
            provider="openai",
            model_name="gpt-test",
            display_name="GPT Test",
            description="Synthetic model",
            dollars_per_million_tokens_input=1.0,
            dollars_per_million_tokens_output=2.0,
            dollars_per_million_tokens_cached_input=0.25,
            dollars_per_million_tokens_cache_write_5m=0.5,
            dollars_per_million_tokens_cache_write_1h=0.75,
            dollars_per_million_tokens_cache_read=0.1,
            cache_storage_cost_per_million_tokens_per_hour=0.01,
            supports_long_context_pricing=True,
            long_context_threshold_tokens=128000,
            dollars_per_million_tokens_input_long_context=0.4,
            dollars_per_million_tokens_output_long_context=1.0,
            supports_thinking=True,
            dollars_per_million_tokens_output_thinking=3.0,
            is_reasoning_model=True,
            min_recommended_reasoning_tokens=2000,
            supports_caching=True,
            supports_streaming=True,
            is_active=True,
        )

        assert model.dollars_per_million_tokens_cached_input == 0.25
        assert model.dollars_per_million_tokens_cache_write_5m == 0.5
        assert model.dollars_per_million_tokens_cache_write_1h == 0.75
        assert model.dollars_per_million_tokens_cache_read == 0.1
        assert model.cache_storage_cost_per_million_tokens_per_hour == 0.01
        assert model.supports_long_context_pricing is True
        assert model.long_context_threshold_tokens == 128000
        assert model.dollars_per_million_tokens_input_long_context == 0.4
        assert model.dollars_per_million_tokens_output_long_context == 1.0
        assert model.supports_thinking is True
        assert model.dollars_per_million_tokens_output_thinking == 3.0


@pytest.mark.asyncio
class TestServiceWithRegistry:
    """Test service integration with registry."""

    @pytest.fixture
    def mock_registry_models(self):
        """Create mock registry models."""
        return [
            RegistryModel(
                provider="openai",
                model_name="gpt-4o-mini",
                display_name="GPT-4o Mini",
                description="Efficient model",
                max_input_tokens=128000,
                max_output_tokens=16384,
                supports_vision=True,
                supports_function_calling=True,
                supports_json_mode=True,
                supports_parallel_tool_calls=True,
                dollars_per_million_tokens_input=0.15,
                dollars_per_million_tokens_output=0.60,
                is_active=True,
            ),
            RegistryModel(
                provider="ollama",
                model_name="llama3",
                display_name="Llama 3",
                description="Open model",
                max_input_tokens=8192,
                max_output_tokens=8192,
                supports_vision=False,
                supports_function_calling=True,
                supports_json_mode=False,
                supports_parallel_tool_calls=False,
                dollars_per_million_tokens_input=0.0,
                dollars_per_million_tokens_output=0.0,
                is_active=True,
            ),
        ]

    async def test_calculate_cost(self, mock_registry_models):
        """Test cost calculation from registry data."""
        from llmring.schemas import LLMResponse

        test_lockfile = Path(__file__).parent.parent / "llmring.lock.json"
        service = LLMRing(lockfile_path=str(test_lockfile))

        # Mock the registry fetch
        service._registry_models = {"openai": mock_registry_models}

        # Create a mock response
        response = LLMResponse(
            content="Test",
            model="openai:gpt-4o-mini",
            usage={"prompt_tokens": 1000, "completion_tokens": 500},
        )

        cost = await service.calculate_cost(response)

        assert cost is not None
        assert cost["input_cost"] == (1000 / 1_000_000) * 0.15
        assert cost["output_cost"] == (500 / 1_000_000) * 0.60
        assert cost["total_cost"] == cost["input_cost"] + cost["output_cost"]

        # For reference
        expected_input_cost = 0.00015  # $0.00015
        expected_output_cost = 0.0003  # $0.0003
        expected_total = 0.00045  # $0.00045

        assert abs(cost["input_cost"] - expected_input_cost) < 0.000001
        assert abs(cost["output_cost"] - expected_output_cost) < 0.000001
        assert abs(cost["total_cost"] - expected_total) < 0.000001

    async def test_calculate_cost_free_model(self, mock_registry_models):
        """Test cost calculation for free models (Ollama)."""
        from llmring.schemas import LLMResponse

        test_lockfile = Path(__file__).parent.parent / "llmring.lock.json"
        service = LLMRing(lockfile_path=str(test_lockfile))

        # Mock the registry fetch
        service._registry_models = {"ollama": mock_registry_models}

        response = LLMResponse(
            content="Test",
            model="ollama:llama3",
            usage={"prompt_tokens": 1000, "completion_tokens": 500},
        )

        cost = await service.calculate_cost(response)

        assert cost is not None
        assert cost["input_cost"] == 0.0
        assert cost["output_cost"] == 0.0
        assert cost["total_cost"] == 0.0

    async def test_validate_context_limit(self, mock_registry_models):
        """Test context limit validation."""
        test_lockfile = Path(__file__).parent.parent / "llmring.lock.json"
        service = LLMRing(lockfile_path=str(test_lockfile))

        # Mock the registry fetch
        service._registry_models = {"openai": mock_registry_models}

        # Create request within limits
        request = LLMRequest(
            messages=[Message(role="user", content="Hello")],
            model="openai:gpt-4o-mini",
            max_tokens=100,
        )

        error = await service.validate_context_limit(request)
        assert error is None  # No error, within limits

        # Create request exceeding limits
        # Create varied content that will exceed 128k token limit
        import string

        varied_content = "".join(
            [string.ascii_letters[i % 52] + str(i % 10) for i in range(550_000)]
        )
        request = LLMRequest(
            messages=[Message(role="user", content=varied_content)],
            model="openai:gpt-4o-mini",
            max_tokens=10000,
        )

        error = await service.validate_context_limit(request)
        assert error is not None
        assert "exceeds" in error
        assert "input limit" in error

    async def test_get_enhanced_model_info(self, mock_registry_models):
        """Test getting enhanced model info with registry data."""
        test_lockfile = Path(__file__).parent.parent / "llmring.lock.json"
        service = LLMRing(lockfile_path=str(test_lockfile))

        # Mock the registry fetch
        service._registry_models = {"openai": mock_registry_models}

        # Get enhanced info
        info = await service.get_enhanced_model_info("openai:gpt-4o-mini")

        assert info["model"] == "gpt-4o-mini"
        assert info["provider"] == "openai"
        assert info["display_name"] == "GPT-4o Mini"
        assert info["max_input_tokens"] == 128000
        assert info["dollars_per_million_tokens_input"] == 0.15
        assert info["supports_vision"] is True

    async def test_enhanced_model_info_missing_registry(self, tmp_path):
        """Test enhanced model info when registry is unavailable."""
        # Use a non-existent registry URL to simulate unavailable registry
        test_lockfile = Path(__file__).parent.parent / "llmring.lock.json"
        service = LLMRing(
            registry_url="file:///non/existent/path", lockfile_path=str(test_lockfile)
        )

        # Clear any cached registry models from previous tests
        service._registry_models = {}
        # Also clear the registry client's cache
        service.registry._cache = {}
        # And use a temp directory for file cache
        service.registry.cache_dir = tmp_path / "cache"
        service.registry.cache_dir.mkdir(exist_ok=True)

        # Don't mock registry - it should handle missing data gracefully
        info = await service.get_enhanced_model_info("openai:gpt-4")

        # Should still have basic info
        assert info["model"] == "gpt-4"
        assert info["provider"] == "openai"
        assert info["supported"] is True  # From provider validation

        # Registry fields will be missing, that's ok
        assert "display_name" not in info or info["display_name"] is None


class TestReasoningModelFields:
    """Test reasoning model specific fields in RegistryModel schema."""

    def test_registry_model_supports_reasoning_fields(self):
        """Test that RegistryModel schema includes reasoning model fields."""
        # Create a reasoning model with the new fields
        reasoning_model = RegistryModel(
            provider="openai",
            model_name="o1",
            display_name="o1",
            description="Reasoning model",
            is_reasoning_model=True,
            min_recommended_reasoning_tokens=5000,
            max_input_tokens=128000,
            max_output_tokens=32768,
            is_active=True,
        )

        # Verify the fields are accessible
        assert reasoning_model.is_reasoning_model is True
        assert reasoning_model.min_recommended_reasoning_tokens == 5000

    def test_registry_model_reasoning_fields_default_values(self):
        """Test default values for reasoning model fields."""
        # Create a non-reasoning model without specifying reasoning fields
        non_reasoning_model = RegistryModel(
            provider="openai",
            model_name="gpt-4",
            display_name="GPT-4",
            description="Non-reasoning model",
            is_active=True,
        )

        # Verify defaults
        assert non_reasoning_model.is_reasoning_model is False
        assert non_reasoning_model.min_recommended_reasoning_tokens is None

    def test_registry_model_parses_reasoning_fields_from_json(self):
        """Test that reasoning fields are preserved when parsing from JSON."""
        # Simulate JSON data from registry
        json_data = {
            "provider": "openai",
            "model_name": "gpt-5",
            "display_name": "GPT-5",
            "description": "GPT-5 reasoning model",
            "is_reasoning_model": True,
            "min_recommended_reasoning_tokens": 2000,
            "max_input_tokens": 272000,
            "max_output_tokens": 128000,
            "is_active": True,
        }

        # Parse the model
        model = RegistryModel(**json_data)

        # Verify reasoning fields are preserved
        assert model.is_reasoning_model is True
        assert model.min_recommended_reasoning_tokens == 2000
        assert model.model_name == "gpt-5"
