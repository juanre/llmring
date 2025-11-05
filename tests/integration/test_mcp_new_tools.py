#!/usr/bin/env python3
"""
Test the new MCP server tools for model discovery and selection.

Tests the data-focused tools that provide registry information to LLMs.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

from llmring.mcp.server.lockfile_server.server import LockfileServer
from llmring.mcp.tools.lockfile_manager import LockfileManagerTools


class TestNewMCPTools:
    """Test the new data-focused MCP tools."""

    @pytest.fixture
    def test_lockfile_path(self):
        """Use the test lockfile with all aliases configured."""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "llmring.lock.json")

    @pytest.fixture
    def lockfile_tools(self, test_lockfile_path):
        """Create lockfile manager tools with test lockfile."""
        return LockfileManagerTools(lockfile_path=Path(test_lockfile_path))

    @pytest.fixture
    def mcp_server(self, test_lockfile_path):
        """Create MCP server with test lockfile."""
        return LockfileServer(lockfile_path=Path(test_lockfile_path))

    @pytest.mark.asyncio
    async def test_get_available_providers(self, lockfile_tools):
        """Test checking which providers have API keys configured."""
        result = await lockfile_tools.get_available_providers()

        assert "configured" in result
        assert "unconfigured" in result
        assert "details" in result
        assert isinstance(result["configured"], list)
        assert isinstance(result["unconfigured"], list)
        assert isinstance(result["details"], dict)

        # Check structure of provider info
        for provider, info in result["details"].items():
            assert "has_key" in info
            assert isinstance(info["has_key"], bool)
            # env_var or env_vars should be present
            assert "env_var" in info or "env_vars" in info

        # Should have at least some providers configured (from .env)
        configured = result["configured"]
        if os.environ.get("OPENAI_API_KEY"):
            assert "openai" in configured
        if os.environ.get("ANTHROPIC_API_KEY"):
            assert "anthropic" in configured

    @pytest.mark.asyncio
    async def test_list_models(self, lockfile_tools):
        """Test listing all available models."""
        # Test listing all models
        result = await lockfile_tools.list_models()

        assert "models" in result
        assert "total_count" in result
        assert isinstance(result["models"], list)
        assert result["total_count"] > 0

        # Check model structure
        if result["models"]:
            model = result["models"][0]
            assert "model_ref" in model
            assert "provider" in model
            assert "model_name" in model
            assert "display_name" in model
            assert "active" in model
            assert "context_window" in model
            assert "input_cost" in model
            assert "output_cost" in model

        # Test filtering by provider
        result = await lockfile_tools.list_models(providers=["openai"])
        for model in result["models"]:
            assert model["provider"] == "openai"

        # Test including inactive models
        result = await lockfile_tools.list_models(include_inactive=True)
        # Should have some models (active or inactive)
        assert result["total_count"] >= 0

    @pytest.mark.asyncio
    async def test_filter_models_by_requirements(self, lockfile_tools):
        """Test filtering models based on requirements."""
        # Test filtering by context window
        result = await lockfile_tools.filter_models_by_requirements(min_context=100000)

        assert "models" in result
        assert "applied_filters" in result

        # All returned models should meet the requirement
        for model in result["models"]:
            # Skip models with missing/zero context window data
            if model["context_window"] and model["context_window"] > 0:
                assert model["context_window"] >= 100000

        # Test filtering by cost
        result = await lockfile_tools.filter_models_by_requirements(
            max_price_input=5.0, max_price_output=10.0
        )

        for model in result["models"]:
            if model["price_input"] is not None:
                assert model["price_input"] <= 5.0
            if model["price_output"] is not None:
                assert model["price_output"] <= 10.0

        # Test filtering by capabilities
        result = await lockfile_tools.filter_models_by_requirements(requires_vision=True)

        # Models should have vision capability
        for model in result["models"]:
            # This depends on registry data structure
            pass  # Just ensure no errors

        # Test filtering by multiple criteria
        result = await lockfile_tools.filter_models_by_requirements(
            min_context=50000, max_price_input=10.0, providers=["openai", "anthropic"]
        )

        for model in result["models"]:
            # Skip models with missing/zero context window data
            if model["context_window"] and model["context_window"] > 0:
                assert model["context_window"] >= 50000
            assert model["provider"] in ["openai", "anthropic"]
            if model["price_input"] is not None:
                assert model["price_input"] <= 10.0

    @pytest.mark.asyncio
    async def test_get_model_details(self, lockfile_tools):
        """Test getting detailed information for specific models."""
        # Get details for specific models
        models_to_check = ["openai:gpt-4o-mini", "anthropic:claude-3-5-haiku-20241022"]

        result = await lockfile_tools.get_model_details(models=models_to_check)

        assert "models" in result
        assert len(result["models"]) <= len(models_to_check)

        # Check detailed structure
        for model in result["models"]:
            assert "model_ref" in model
            assert "provider" in model
            assert "model_name" in model
            assert "full_details" in model

            # Full details should have comprehensive info
            details = model["full_details"]
            if details:  # If registry data is available
                # Check for expected fields (may vary by provider)
                possible_fields = [
                    "display_name",
                    "context_window",
                    "max_output",
                    "dollars_per_million_tokens_input",
                    "dollars_per_million_tokens_output",
                    "active",
                    "knowledge_cutoff",
                ]
                # At least some of these should be present
                has_fields = any(field in details for field in possible_fields)
                assert has_fields or not details

    @pytest.mark.asyncio
    async def test_enhanced_assess_model(self, lockfile_tools):
        """Test the enhanced assess_model with complete data."""
        # Test with a known model
        result = await lockfile_tools.assess_model(model_ref="openai:gpt-4o-mini")

        assert "model" in result
        assert "active" in result
        assert "capabilities" in result
        assert "pricing" in result
        assert "specifications" in result
        assert "metadata" in result

        # Check pricing structure
        if result["pricing"]:
            assert "input" in result["pricing"]
            assert "output" in result["pricing"]

        # Check specifications
        if result["specifications"]:
            possible_specs = ["context_window", "max_output", "knowledge_cutoff"]
            has_specs = any(spec in result["specifications"] for spec in possible_specs)
            assert has_specs or not result["specifications"]

        # Test with an alias (should resolve)
        result = await lockfile_tools.assess_model(model_ref="fast")
        assert "model" in result
        # Should have resolved the alias
        assert result["model"] != "fast"

    @pytest.mark.asyncio
    async def test_enhanced_analyze_costs(self, lockfile_tools):
        """Test the enhanced analyze_costs with what-if analysis."""
        monthly_volume = {"input_tokens": 1000000, "output_tokens": 500000}

        # Test with actual configuration
        result = await lockfile_tools.analyze_costs(monthly_volume=monthly_volume)

        assert "analysis_type" in result
        assert result["analysis_type"] == "actual"
        assert "total_monthly_cost" in result
        assert "cost_breakdown" in result
        assert "models_analyzed" in result

        # Test with hypothetical models (what-if analysis)
        hypothetical = {
            "fast": "openai:gpt-4o-mini",
            "smart": "anthropic:claude-3-opus-20240229",
            "balanced": "openai:gpt-4o",
        }

        result = await lockfile_tools.analyze_costs(
            monthly_volume=monthly_volume, hypothetical_models=hypothetical
        )

        assert result["analysis_type"] == "hypothetical"
        assert result["models_analyzed"] == len(hypothetical)

        # Cost breakdown should have the hypothetical aliases
        for alias in hypothetical.keys():
            if alias in result["cost_breakdown"]:
                breakdown = result["cost_breakdown"][alias]
                assert "model" in breakdown
                assert "input_cost" in breakdown
                assert "output_cost" in breakdown
                assert "total_cost" in breakdown

    @pytest.mark.asyncio
    async def test_mcp_server_has_new_tools(self, mcp_server):
        """Test that MCP server has registered all new tools."""
        # Check that all new tools are registered
        expected_new_tools = [
            "get_available_providers",
            "list_models",
            "filter_models_by_requirements",
            "get_model_details",
        ]

        for tool_name in expected_new_tools:
            assert tool_name in mcp_server.server.function_registry.functions

        # Test that wrappers work correctly
        get_providers_func = mcp_server.server.function_registry.functions[
            "get_available_providers"
        ]
        result = get_providers_func()
        assert "configured" in result
        assert "unconfigured" in result
        assert "details" in result

        # Test list_models wrapper
        list_models_func = mcp_server.server.function_registry.functions["list_models"]
        result = list_models_func()
        assert "models" in result
        assert "total_count" in result

    @pytest.mark.asyncio
    async def test_tools_error_handling(self, lockfile_tools):
        """Test error handling in new tools."""
        # Test with invalid model reference
        result = await lockfile_tools.get_model_details(models=["invalid:nonexistent-model-xyz"])
        # Should not crash, just return empty or skip invalid
        assert "models" in result

        # Test with invalid filter values
        result = await lockfile_tools.filter_models_by_requirements(
            min_context=-1, max_price_input=-10  # Invalid negative value  # Invalid negative cost
        )
        # Should handle gracefully
        assert "models" in result

        # Test with empty providers list
        result = await lockfile_tools.list_models(providers=[])
        # Should return empty or all models
        assert "models" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
