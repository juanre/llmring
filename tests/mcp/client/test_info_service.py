"""
Test Information Service functionality.

This test module verifies that the information service provides comprehensive
transparency into providers, models, costs, usage, and data storage.
"""

from unittest.mock import Mock

import pytest

from llmring.mcp.client.enhanced_llm import create_enhanced_llm
from llmring.mcp.client.info_service import DataStorageInfo, create_info_service


class TestInfoServiceBasics:
    """Test basic information service functionality."""

    def test_creation(self):
        """Test info service creation."""
        info_service = create_info_service(origin="test-info")

        assert info_service is not None
        assert info_service.origin == "test-info"
        assert info_service.llm_service is None
        assert info_service.llm_db is None

    def test_creation_with_params(self):
        """Test info service creation with parameters."""
        mock_llm_service = Mock()

        info_service = create_info_service(llm_service=mock_llm_service, origin="test-module")

        assert info_service.llm_service == mock_llm_service
        assert info_service.origin == "test-module"


class TestProviderInformation:
    """Test provider information functionality."""

    @pytest.mark.asyncio
    async def test_get_available_providers_no_service(self):
        """Test getting providers when no LLM service is available."""
        info_service = create_info_service()
        providers = await info_service.get_available_providers()

        assert isinstance(providers, list)
        assert len(providers) == 0

    @pytest.mark.asyncio
    async def test_get_available_providers_with_service(self):
        """Test getting providers with LLM service."""
        from unittest.mock import AsyncMock

        mock_llm_service = Mock()
        mock_llm_service.get_available_models = AsyncMock(
            return_value={
                "anthropic": ["claude-3-sonnet", "claude-3-haiku"],
                "openai": ["gpt-4o", "gpt-4o-mini"],
                "google": ["gemini-pro"],
                "ollama": ["llama3.1"],
            }
        )
        mock_llm_service.providers = {"anthropic": True, "openai": True}

        info_service = create_info_service(llm_service=mock_llm_service)
        providers = await info_service.get_available_providers()

        assert isinstance(providers, list)
        assert len(providers) == 4

        # Check provider structure
        anthropic_provider = next(p for p in providers if p.name == "anthropic")
        assert anthropic_provider.available is True
        assert anthropic_provider.model_count == 2
        assert anthropic_provider.api_key_configured is True
        assert "Claude" in anthropic_provider.description

        # Check unavailable provider
        google_provider = next(p for p in providers if p.name == "google")
        assert google_provider.available is False
        assert google_provider.api_key_configured is False


class TestModelInformation:
    """Test model information functionality."""

    @pytest.mark.asyncio
    async def test_get_models_for_provider_no_db(self):
        """Test getting models when no database is available."""
        from unittest.mock import AsyncMock

        mock_llm_service = Mock()
        mock_llm_service.get_available_models = AsyncMock(
            return_value={"anthropic": ["claude-3-sonnet", "claude-3-haiku"]}
        )

        info_service = create_info_service(llm_service=mock_llm_service)
        models = await info_service.get_models_for_provider("anthropic")

        assert isinstance(models, list)
        assert len(models) == 2

        # Check fallback model structure (minimal info)
        sonnet_model = next(m for m in models if m.model_name == "claude-3-sonnet")
        assert sonnet_model.provider == "anthropic"
        assert sonnet_model.display_name == "claude-3-sonnet"
        assert sonnet_model.cost_per_input_token is None
        assert sonnet_model.supports_function_calling is False

    @pytest.mark.asyncio
    async def test_get_models_for_provider_with_db(self):
        """Test getting models with database information."""
        # Create mock database
        mock_db = Mock()
        mock_model = Mock()
        mock_model.provider = "anthropic"
        mock_model.model_name = "claude-3-sonnet"
        mock_model.display_name = "Claude 3 Sonnet"
        mock_model.description = "Anthropic's Claude 3 Sonnet model"
        mock_model.max_context = 200000
        mock_model.max_output_tokens = 4096
        mock_model.supports_vision = True
        mock_model.supports_function_calling = True
        mock_model.supports_json_mode = True
        mock_model.supports_parallel_tool_calls = True
        mock_model.cost_per_token_input = 0.000003  # Actual Claude Sonnet cost per token
        mock_model.cost_per_token_output = 0.000015  # Actual Claude Sonnet cost per token
        mock_model.is_active = True

        mock_db.list_models.return_value = [mock_model]

        info_service = create_info_service()
        info_service.llm_db = mock_db

        models = await info_service.get_models_for_provider("anthropic")

        assert len(models) == 1
        model = models[0]
        assert model.provider == "anthropic"
        assert model.model_name == "claude-3-sonnet"
        assert model.display_name == "Claude 3 Sonnet"
        assert model.max_context_tokens == 200000
        assert model.supports_vision is True
        assert model.supports_function_calling is True
        assert model.cost_per_1k_input_tokens == 0.003  # 0.000003 * 1000
        assert model.cost_per_1k_output_tokens == 0.015  # 0.000015 * 1000

    @pytest.mark.asyncio
    async def test_get_model_cost_info(self):
        """Test getting cost information for a specific model."""
        mock_db = Mock()
        mock_model = Mock()
        mock_model.provider = "anthropic"
        mock_model.model_name = "claude-3-sonnet"
        mock_model.display_name = "Claude 3 Sonnet"
        mock_model.cost_per_token_input = 0.000003  # Actual Claude Sonnet cost per token
        mock_model.cost_per_token_output = 0.000015  # Actual Claude Sonnet cost per token
        mock_model.max_context = 200000
        mock_model.supports_function_calling = True
        mock_model.supports_vision = True
        mock_model.description = "Test model"
        mock_model.max_output_tokens = 4096
        mock_model.supports_json_mode = True
        mock_model.supports_parallel_tool_calls = True
        mock_model.is_active = True

        mock_db.list_models.return_value = [mock_model]

        info_service = create_info_service()
        info_service.llm_db = mock_db

        # Test with provider:model format
        cost_info = await info_service.get_model_cost_info("anthropic:claude-3-sonnet")

        assert cost_info is not None
        assert cost_info["provider"] == "anthropic"
        assert cost_info["model_name"] == "claude-3-sonnet"
        assert cost_info["cost_per_input_token"] == 0.000003
        assert cost_info["cost_per_output_token"] == 0.000015
        assert cost_info["cost_per_1k_input_tokens"] == 0.003
        assert cost_info["cost_per_1k_output_tokens"] == 0.015
        assert cost_info["supports_function_calling"] is True

        # Test with unknown model
        unknown_cost = await info_service.get_model_cost_info("unknown:model")
        assert unknown_cost is None


class TestUsageStatistics:
    """Test usage statistics functionality."""

    def test_get_usage_stats_no_service(self):
        """Test getting usage stats when no LLM service is available."""
        info_service = create_info_service()
        stats = info_service.get_usage_stats("test-user")

        assert stats is None

    def test_get_usage_stats_with_service(self):
        """Test getting usage stats with LLM service."""
        mock_llm_service = Mock()
        mock_stats = Mock()
        mock_stats.total_calls = 50
        mock_stats.total_tokens = 10000
        mock_stats.total_input_tokens = 7000
        mock_stats.total_output_tokens = 3000
        mock_stats.total_cost = 0.15
        mock_stats.avg_cost_per_call = 0.003
        mock_stats.most_used_model = "anthropic:claude-3-sonnet"
        mock_stats.success_rate = 0.98
        mock_stats.avg_response_time_ms = 1500

        mock_llm_service.get_usage_stats.return_value = mock_stats

        info_service = create_info_service(llm_service=mock_llm_service)
        stats = info_service.get_usage_stats("test-user", days=30)

        assert stats is not None
        assert stats.user_id == "test-user"
        assert stats.total_calls == 50
        assert stats.total_tokens == 10000
        assert stats.total_cost == 0.15
        assert stats.most_used_model == "anthropic:claude-3-sonnet"
        assert stats.success_rate == 0.98


class TestDataStorageInformation:
    """Test data storage information functionality."""

    def test_get_data_storage_info(self):
        """Test getting data storage information."""
        info_service = create_info_service()
        storage_info = info_service.get_data_storage_info()

        assert isinstance(storage_info, DataStorageInfo)
        assert len(storage_info.mcp_client_tables) > 0
        assert len(storage_info.llm_service_tables) > 0
        assert len(storage_info.user_data_locations) > 0
        assert len(storage_info.retention_policies) > 0
        assert len(storage_info.privacy_measures) > 0

        # Check specific HTTP endpoints are documented (no direct database access)
        table_names = [table["table"] for table in storage_info.mcp_client_tables]
        assert "conversations" in table_names
        assert "messages" in table_names
        assert "mcp_servers" in table_names
        assert "conversation_templates" in table_names

        llm_table_names = [table["table"] for table in storage_info.llm_service_tables]
        assert "usage_logs" in llm_table_names

        # Check user data locations
        assert "conversation_content" in storage_info.user_data_locations
        assert "llm_usage_tracking" in storage_info.user_data_locations

        # Check privacy measures
        privacy_text = " ".join(storage_info.privacy_measures).lower()
        assert "sha-256" in privacy_text
        assert "isolation" in privacy_text
        assert "user_id" in privacy_text

    def test_get_user_data_summary(self):
        """Test getting user data summary."""
        info_service = create_info_service(origin="test-module")
        summary = info_service.get_user_data_summary("test-user")

        assert summary["user_id"] == "test-user"
        assert summary["origin"] == "test-module"
        assert "data_locations" in summary
        assert "privacy_info" in summary

        # Check privacy information
        privacy_info = summary["privacy_info"]
        assert privacy_info["system_prompts_hashed"] is True
        assert "user_messages_stored" in privacy_info
        assert "data_retention" in privacy_info
        assert "data_isolation" in privacy_info


class TestEnhancedLLMIntegration:
    """Test integration with Enhanced LLM."""

    @pytest.mark.asyncio
    async def test_enhanced_llm_transparency_methods(self):
        """Test transparency methods in Enhanced LLM."""
        enhanced_llm = create_enhanced_llm(origin="test-integration")

        try:
            # Test provider information
            providers = await enhanced_llm.get_available_providers()
            assert isinstance(providers, list)

            # Test data storage information
            storage_info = enhanced_llm.get_data_storage_info()
            assert "mcp_client_tables" in storage_info
            assert "llm_service_tables" in storage_info

            # Test user data summary
            user_summary = enhanced_llm.get_user_data_summary()
            assert "user_id" in user_summary
            assert "origin" in user_summary

            # Test comprehensive transparency report
            report = await enhanced_llm.get_transparency_report()
            assert "user_id" in report
            assert "generated_at" in report
            assert "enhanced_llm_config" in report
            assert "available_providers" in report
            assert "data_storage" in report
            assert "user_data_summary" in report

            # Check enhanced LLM specific information
            config = report["enhanced_llm_config"]
            assert config["origin"] == "test-integration"
            assert "default_model" in config
            assert "registered_tools" in config
            assert "mcp_server_connected" in config

        finally:
            await enhanced_llm.close()

    @pytest.mark.asyncio
    async def test_enhanced_llm_with_registered_tools(self):
        """Test transparency with registered tools."""
        enhanced_llm = create_enhanced_llm(origin="test-tools")

        try:
            # Register a test tool
            def test_calculator(expression: str) -> float:
                return eval(expression.replace("^", "**"))

            enhanced_llm.register_tool(
                name="calculator",
                description="Perform calculations",
                parameters={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
                handler=test_calculator,
                module_name="math-module",
            )

            # Get transparency report
            report = await enhanced_llm.get_transparency_report()

            # Check that registered tools are included
            config = report["enhanced_llm_config"]
            assert len(config["registered_tools"]) == 1

            tool_info = config["registered_tools"][0]
            assert tool_info["name"] == "calculator"
            assert tool_info["description"] == "Perform calculations"
            assert tool_info["module_name"] == "math-module"

        finally:
            await enhanced_llm.close()


class TestInfoServiceSerialization:
    """Test serialization functionality."""

    def test_to_dict_conversion(self):
        """Test converting objects to dictionaries."""
        info_service = create_info_service()

        # Test with a simple object
        test_obj = Mock()
        test_obj.__dict__ = {"name": "test", "count": 42, "nested": Mock()}
        test_obj.nested.__dict__ = {"value": "nested_value"}

        result = info_service.to_dict(test_obj)

        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["count"] == 42
        assert isinstance(result["nested"], dict)
        assert result["nested"]["value"] == "nested_value"

    def test_data_storage_info_serialization(self):
        """Test that DataStorageInfo can be serialized."""
        info_service = create_info_service()
        storage_info = info_service.get_data_storage_info()

        # Convert to dict
        storage_dict = info_service.to_dict(storage_info)

        assert isinstance(storage_dict, dict)
        assert "mcp_client_tables" in storage_dict
        assert "llm_service_tables" in storage_dict
        assert "user_data_locations" in storage_dict
        assert "retention_policies" in storage_dict
        assert "privacy_measures" in storage_dict

        # Verify structure is preserved
        assert isinstance(storage_dict["mcp_client_tables"], list)
        assert isinstance(storage_dict["user_data_locations"], dict)


class TestAdvancedFeatures:
    """Test advanced information service features."""

    def test_provider_usage_stats_integration(self):
        """Test provider usage statistics integration."""
        mock_llm_service = Mock()
        mock_stats = Mock()
        mock_stats.total_cost = 0.25
        mock_stats.total_calls = 100
        mock_llm_service.get_usage_stats.return_value = mock_stats

        mock_db = Mock()
        mock_model = Mock()
        mock_model.model_name = "claude-3-sonnet"
        mock_db.list_models.return_value = [mock_model]

        info_service = create_info_service(llm_service=mock_llm_service)
        info_service.llm_db = mock_db

        # Test internal provider stats method
        stats = info_service._get_provider_usage_stats("anthropic")

        if stats:  # May be None depending on implementation
            assert "total_cost" in stats
            assert "total_calls" in stats
            assert stats["provider"] == "anthropic"

    def test_model_usage_stats_integration(self):
        """Test model usage statistics integration."""
        mock_llm_service = Mock()
        mock_stats = Mock()
        mock_stats.total_cost = 0.15
        mock_stats.total_tokens = 5000
        mock_stats.total_calls = 25
        mock_stats.most_used_model = "anthropic:claude-3-sonnet"
        mock_llm_service.get_usage_stats.return_value = mock_stats

        info_service = create_info_service(llm_service=mock_llm_service)

        # Test internal model stats method
        stats = info_service._get_model_usage_stats("anthropic", "claude-3-sonnet")

        assert stats is not None
        assert stats["model"] == "anthropic:claude-3-sonnet"
        assert "total_cost" in stats
        assert "total_tokens" in stats

    def test_daily_breakdown_generation(self):
        """Test daily usage breakdown generation."""
        mock_llm_service = Mock()
        mock_stats = Mock()
        mock_stats.total_cost = 0.21  # 7 days * 0.03
        mock_stats.total_tokens = 7000  # 7 days * 1000
        mock_stats.total_calls = 70  # 7 days * 10
        mock_llm_service.get_usage_stats.return_value = mock_stats

        info_service = create_info_service(llm_service=mock_llm_service)

        # Test daily breakdown
        breakdown = info_service._get_daily_usage_breakdown("test-user", 7)

        assert isinstance(breakdown, list)
        assert len(breakdown) == 7  # Should have 7 days

        if breakdown:  # Check structure if data is available
            day_entry = breakdown[0]
            assert "date" in day_entry
            assert "calls" in day_entry
            assert "tokens" in day_entry
            assert "cost" in day_entry
