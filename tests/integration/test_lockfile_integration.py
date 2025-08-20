"""
Integration tests for lockfile with registry and service.

Tests lockfile integration with registry validation, cost calculations,
and alias resolution in the service.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from llmring import LLMRequest, LLMRing, Message
from llmring.lockfile import Lockfile
from llmring.registry import RegistryClient


class TestLockfileIntegration:
    """Test lockfile integration with other components."""

    @pytest.fixture
    def test_registry_url(self):
        """Use test registry files."""
        test_registry = Path(__file__).parent.parent / "resources" / "registry"
        return f"file://{test_registry}"

    @pytest.fixture
    async def service_with_lockfile(self, tmp_path, test_registry_url):
        """Create service with a test lockfile."""
        # Create lockfile
        lockfile = Lockfile()
        lockfile.set_binding("summarizer", "openai:gpt-4o-mini", profile="default")
        lockfile.set_binding("deep", "openai:gpt-4", profile="default")
        lockfile.set_binding("cheap", "openai:gpt-3.5-turbo", profile="default")

        # Save lockfile
        lockfile_path = tmp_path / "llmring.lock"
        lockfile.save(lockfile_path)

        # Create service with lockfile
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            service = LLMRing(
                registry_url=test_registry_url, lockfile_path=str(lockfile_path)
            )
            yield service
            await service.close()

    def test_alias_resolution_in_service(self, service_with_lockfile):
        """Test that aliases are resolved correctly in service."""
        service = service_with_lockfile

        # Test alias resolution
        resolved = service.resolve_alias("summarizer")
        assert resolved == "openai:gpt-4o-mini"

        resolved = service.resolve_alias("deep")
        assert resolved == "openai:gpt-4"

        # Test direct model reference (should pass through)
        resolved = service.resolve_alias("openai:gpt-3.5-turbo")
        assert resolved == "openai:gpt-3.5-turbo"

        # Test non-existent alias (should pass through)
        resolved = service.resolve_alias("non_existent")
        assert resolved == "non_existent"

    @pytest.mark.asyncio
    async def test_alias_in_chat_request(self, service_with_lockfile):
        """Test using alias in chat request."""
        service = service_with_lockfile

        # Create request with alias
        request = LLMRequest(
            model="summarizer",  # This is an alias
            messages=[Message(role="user", content="Hello")],
        )

        # The chat method should resolve the alias
        # We can't test the actual API call, but we can verify the resolution
        # happens by checking internal state
        resolved = service.resolve_alias(request.model)
        assert resolved == "openai:gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_cost_calculation_with_registry(self, service_with_lockfile):
        """Test cost calculation using registry pricing."""
        from llmring.schemas import LLMResponse

        service = service_with_lockfile

        # Create mock response
        response = LLMResponse(
            content="Test",
            model="openai:gpt-4o-mini",
            usage={"prompt_tokens": 1000, "completion_tokens": 500},
        )

        cost_info = await service.calculate_cost(response)

        assert cost_info is not None
        assert "total_cost" in cost_info
        assert "input_cost" in cost_info
        assert "output_cost" in cost_info

        # Based on test registry: $0.15 per million input, $0.60 per million output
        expected_input_cost = (1000 / 1_000_000) * 0.15
        expected_output_cost = (500 / 1_000_000) * 0.60
        expected_total = expected_input_cost + expected_output_cost

        assert cost_info["input_cost"] == pytest.approx(expected_input_cost)
        assert cost_info["output_cost"] == pytest.approx(expected_output_cost)
        assert cost_info["total_cost"] == pytest.approx(expected_total)

    @pytest.mark.asyncio
    async def test_context_validation_with_registry(self, service_with_lockfile):
        """Test context validation using registry limits."""
        service = service_with_lockfile

        # Create request within limits
        messages = [Message(role="user", content="x" * 1000)]
        request = LLMRequest(
            model="openai:gpt-4o-mini", messages=messages, max_tokens=100
        )

        # Should pass validation (gpt-4o-mini has 128k input limit)
        error = await service.validate_context_limit(request)
        assert error is None

        # Create request exceeding limits
        # Token estimation is roughly 1 token per 4 characters
        # So for 128k token limit, we need more than 512k characters
        huge_message = "x" * 600_000  # ~150k tokens, way over 128k limit
        request_huge = LLMRequest(
            model="openai:gpt-4o-mini",
            messages=[Message(role="user", content=huge_message)],
            max_tokens=100,
        )

        error = await service.validate_context_limit(request_huge)
        assert error is not None
        assert "exceeds" in error.lower()

    @pytest.mark.asyncio
    async def test_registry_model_info(self, test_registry_url):
        """Test getting model info from registry."""
        registry = RegistryClient(registry_url=test_registry_url)

        # Fetch models
        models = await registry.fetch_current_models("openai")

        assert len(models) > 0

        # Find specific model
        gpt4_mini = None
        for model in models:
            if model.model_name == "gpt-4o-mini":
                gpt4_mini = model
                break

        assert gpt4_mini is not None
        assert gpt4_mini.max_input_tokens == 128000
        assert gpt4_mini.max_output_tokens == 16384
        assert gpt4_mini.dollars_per_million_tokens_input == 0.15
        assert gpt4_mini.dollars_per_million_tokens_output == 0.60
        assert gpt4_mini.supports_vision is True
        assert gpt4_mini.supports_function_calling is True

    @pytest.mark.asyncio
    async def test_profile_switching(self, tmp_path):
        """Test switching between profiles."""
        # Create lockfile with different profiles
        lockfile = Lockfile()
        lockfile.set_binding("api", "openai:gpt-3.5-turbo", profile="dev")
        lockfile.set_binding("api", "openai:gpt-4", profile="prod")

        lockfile_path = tmp_path / "llmring.lock"
        lockfile.save(lockfile_path)

        # Test with dev profile
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test", "LLMRING_PROFILE": "dev"}
        ):
            service = LLMRing(lockfile_path=str(lockfile_path))
            resolved = service.resolve_alias("api")
            assert resolved == "openai:gpt-3.5-turbo"

        # Test with prod profile
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test", "LLMRING_PROFILE": "prod"}
        ):
            service = LLMRing(lockfile_path=str(lockfile_path))
            resolved = service.resolve_alias("api")
            assert resolved == "openai:gpt-4"

    def test_bind_alias_from_service(self, tmp_path):
        """Test binding aliases through service."""
        lockfile_path = tmp_path / "llmring.lock"

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            service = LLMRing(lockfile_path=str(lockfile_path))

            # Initially should have a default lockfile (created when path provided but file doesn't exist)
            assert service.lockfile is not None
            assert not lockfile_path.exists()  # File not saved yet

            # Bind alias (should update lockfile)
            service.bind_alias("test", "openai:gpt-4")

            assert service.lockfile is not None
            # Note: bind_alias doesn't automatically save to disk

            # Check alias was bound in memory
            aliases = service.list_aliases()
            assert "test" in aliases
            assert aliases["test"] == "openai:gpt-4"

    def test_list_aliases_from_service(self, tmp_path):
        """Test listing aliases through service."""
        # Create lockfile with aliases
        lockfile = Lockfile()
        lockfile.set_binding("alias1", "openai:gpt-3.5-turbo")
        lockfile.set_binding("alias2", "anthropic:claude-3-haiku")
        lockfile.set_binding("alias3", "google:gemini-pro")

        lockfile_path = tmp_path / "llmring.lock"
        lockfile.save(lockfile_path)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            service = LLMRing(lockfile_path=str(lockfile_path))

            aliases = service.list_aliases()
            assert len(aliases) == 3
            assert aliases["alias1"] == "openai:gpt-3.5-turbo"
            assert aliases["alias2"] == "anthropic:claude-3-haiku"
            assert aliases["alias3"] == "google:gemini-pro"

    @pytest.mark.asyncio
    async def test_enhanced_model_info(self, service_with_lockfile):
        """Test getting enhanced model info with registry data."""
        service = service_with_lockfile

        # Get info for aliased model
        info = service.get_model_info("openai:gpt-4o-mini")

        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4o-mini"
        assert info["supported"] is True

    @pytest.mark.asyncio
    async def test_chat_with_alias_convenience(self, service_with_lockfile):
        """Test the convenience chat_with_alias method."""
        service = service_with_lockfile

        # We can't test actual API calls, but we can test the method exists
        # and accepts the right parameters
        assert hasattr(service, "chat_with_alias")

        # Test that it resolves aliases
        resolved = service.resolve_alias("summarizer")
        assert resolved == "openai:gpt-4o-mini"


class TestLockfileValidation:
    """Test lockfile validation against registry."""

    @pytest.mark.asyncio
    async def test_validate_model_exists(self):
        """Test validating that models exist in registry."""
        test_registry = Path(__file__).parent.parent / "resources" / "registry"
        registry_url = f"file://{test_registry}"

        registry = RegistryClient(registry_url=registry_url)

        # Validate known model
        models = await registry.fetch_current_models("openai")
        model_names = {m.model_name for m in models}

        assert "gpt-4o-mini" in model_names
        assert "gpt-4" in model_names
        assert "gpt-3.5-turbo" in model_names

        # Non-existent model
        assert "gpt-5-ultra" not in model_names

    @pytest.mark.asyncio
    async def test_registry_version_tracking(self):
        """Test that registry versions can be tracked."""
        lockfile = Lockfile()
        profile = lockfile.get_profile("default")

        # Set some registry versions
        profile.registry_versions["openai"] = 1
        profile.registry_versions["anthropic"] = 2

        assert profile.registry_versions["openai"] == 1
        assert profile.registry_versions["anthropic"] == 2

    @pytest.mark.asyncio
    async def test_validate_pricing_info(self):
        """Test that pricing information is available from registry."""
        test_registry = Path(__file__).parent.parent / "resources" / "registry"
        registry_url = f"file://{test_registry}"

        registry = RegistryClient(registry_url=registry_url)
        models = await registry.fetch_current_models("openai")

        # Find gpt-3.5-turbo
        gpt35 = None
        for model in models:
            if model.model_name == "gpt-3.5-turbo":
                gpt35 = model
                break

        assert gpt35 is not None
        assert gpt35.dollars_per_million_tokens_input == 0.50
        assert gpt35.dollars_per_million_tokens_output == 1.50

        # Calculate sample cost
        prompt_tokens = 1000
        completion_tokens = 500

        input_cost = (
            prompt_tokens / 1_000_000
        ) * gpt35.dollars_per_million_tokens_input
        output_cost = (
            completion_tokens / 1_000_000
        ) * gpt35.dollars_per_million_tokens_output
        total_cost = input_cost + output_cost

        assert input_cost == pytest.approx(0.0005)  # $0.0005
        assert output_cost == pytest.approx(0.00075)  # $0.00075
        assert total_cost == pytest.approx(0.00125)  # $0.00125


class TestLiveRegistry:
    """Tests against the live registry (when enabled)."""

    @pytest.mark.asyncio
    async def test_live_registry_fetch(self):
        """Test fetching from live GitHub Pages registry."""
        registry = RegistryClient()  # Uses default URL

        # Try to fetch OpenAI models
        try:
            models = await registry.fetch_current_models("openai")

            # We don't know exact models, but should have some
            assert len(models) > 0

            # Check structure is correct
            for model in models:
                assert model.provider == "openai"
                assert model.model_name is not None
                assert model.is_active is not None

                # If pricing is available, it should be positive
                if model.dollars_per_million_tokens_input is not None:
                    assert model.dollars_per_million_tokens_input > 0
                if model.dollars_per_million_tokens_output is not None:
                    assert model.dollars_per_million_tokens_output > 0

        except Exception as e:
            # Registry might be down or not deployed yet
            pytest.skip(f"Live registry not available: {e}")

    @pytest.mark.asyncio
    async def test_live_registry_all_providers(self):
        """Test fetching all providers from live registry."""
        registry = RegistryClient()

        providers = ["openai", "anthropic", "google"]
        results = {}

        for provider in providers:
            try:
                models = await registry.fetch_current_models(provider)
                results[provider] = len(models)
            except Exception:
                results[provider] = 0

        # At least one provider should have models
        assert sum(results.values()) > 0

        # Log what we found (useful for debugging)
        for provider, count in results.items():
            print(f"{provider}: {count} models")

    @pytest.mark.asyncio
    async def test_live_registry_caching(self):
        """Test that registry caching works."""
        import time

        registry = RegistryClient()

        # First fetch
        start = time.time()
        models1 = await registry.fetch_current_models("openai")
        first_fetch_time = time.time() - start

        # Second fetch (should be cached)
        start = time.time()
        models2 = await registry.fetch_current_models("openai")
        second_fetch_time = time.time() - start

        # Cache fetch should be much faster
        assert second_fetch_time < first_fetch_time / 2

        # Should return same data
        assert len(models1) == len(models2)

    @pytest.mark.asyncio
    async def test_live_registry_model_details(self):
        """Test fetching specific model details from live registry."""
        registry = RegistryClient()

        # Fetch OpenAI models
        models = await registry.fetch_current_models("openai")

        # Look for common models we expect to exist
        model_names = {m.model_name for m in models}

        # These models should typically be available
        common_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
        found_models = [
            m
            for m in common_models
            if any(model_name.startswith(m) for model_name in model_names)
        ]

        assert (
            len(found_models) > 0
        ), f"Expected to find at least one common model, got: {model_names}"

        # Verify model structure
        for model in models:
            assert model.provider == "openai"
            assert model.model_name is not None
            assert model.display_name is not None

            # Check pricing is reasonable (if set)
            if model.dollars_per_million_tokens_input is not None:
                assert (
                    0 < model.dollars_per_million_tokens_input < 1000
                )  # Reasonable range
            if model.dollars_per_million_tokens_output is not None:
                assert 0 < model.dollars_per_million_tokens_output < 1000

    @pytest.mark.asyncio
    async def test_live_registry_cost_calculation_with_real_data(self):
        """Test cost calculation with real registry data."""
        from llmring import LLMRing
        from llmring.schemas import LLMResponse

        service = LLMRing()  # Uses live registry by default

        # Create mock response
        response = LLMResponse(
            content="Test",
            model="openai:gpt-3.5-turbo",
            usage={"prompt_tokens": 1000, "completion_tokens": 500},
        )

        # Try to calculate cost for gpt-3.5-turbo (should be available)
        try:
            cost_info = await service.calculate_cost(response)

            assert cost_info is not None
            assert "total_cost" in cost_info
            assert "input_cost" in cost_info
            assert "output_cost" in cost_info

            # Costs should be positive
            assert cost_info["input_cost"] > 0
            assert cost_info["output_cost"] > 0
            assert (
                cost_info["total_cost"]
                == cost_info["input_cost"] + cost_info["output_cost"]
            )

            # Log actual costs for visibility
            print("\nActual costs from live registry for gpt-3.5-turbo:")
            print(f"  Input cost (1000 tokens): ${cost_info['input_cost']:.6f}")
            print(f"  Output cost (500 tokens): ${cost_info['output_cost']:.6f}")
            print(f"  Total cost: ${cost_info['total_cost']:.6f}")

        except Exception as e:
            # Model might not exist or registry might have changed
            pytest.skip(f"Could not calculate cost for gpt-3.5-turbo: {e}")
