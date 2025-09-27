#!/usr/bin/env python3
"""
Real integration tests for MCP lockfile management without mocks.

Tests the complete flow from chat app to server to tools with actual execution.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

from llmring.lockfile_core import Lockfile
from llmring.mcp.client.chat.app import MCPChatApp
from llmring.mcp.server.lockfile_server.server import LockfileServer
from llmring.mcp.server.transport.stdio import StdioTransport
from llmring.mcp.tools.lockfile_manager import LockfileManagerTools
from llmring.schemas import LLMRequest, LLMResponse, Message


class RealMCPTestEnvironment:
    """Real test environment with actual MCP server and client."""

    def __init__(self, lockfile_path: Path = None):
        """Initialize test environment."""
        self.lockfile_path = lockfile_path or Path("test_llmring.lock")
        self.server = None
        self.client = None
        self.server_task = None

    async def start_server(self):
        """Start the real MCP server in background."""
        self.server = LockfileServer(lockfile_path=self.lockfile_path)

        # Start server in background task
        import subprocess
        import sys

        # Start server as subprocess for real stdio communication
        self.server_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "llmring.mcp.server.lockfile_server",
                "--lockfile",
                str(self.lockfile_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,  # Binary mode for proper MCP protocol
        )

        # Give server time to start
        await asyncio.sleep(0.5)

    async def start_client(self):
        """Start the real MCP client chat app."""
        # Use the actual server process
        self.client = MCPChatApp(
            mcp_server_url="stdio://python -m llmring.mcp.server.lockfile_server",
            llm_model="advisor",  # Use the advisor alias
            enable_telemetry=False,  # Disable for tests
        )

        await self.client.initialize_async()

    async def cleanup(self):
        """Clean up resources."""
        if self.client:
            await self.client.cleanup()

        if hasattr(self, "server_process"):
            self.server_process.terminate()
            await asyncio.sleep(0.1)
            self.server_process.kill()


@pytest.mark.asyncio
async def test_real_mcp_tool_execution():
    """Test real MCP tool execution without any mocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "test.lock"

        # Direct tool testing first
        # Load test lockfile to get real models
        test_lockfile_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "llmring.lock.json"
        )
        test_lockfile = Lockfile.load(Path(test_lockfile_path))

        tools = LockfileManagerTools(lockfile_path=lockfile_path)

        # Test add_alias directly - resolve model from test lockfile
        resolved_model = test_lockfile.resolve_alias("fast")
        result = await tools.add_alias(alias="test_fast", model=resolved_model)

        assert result["success"] is True
        assert result["alias"] == "test_fast"
        assert "model" in result

        # Test list_aliases
        list_result = await tools.list_aliases()
        assert len(list_result["aliases"]) > 0

        # Find our added alias
        found = False
        for alias in list_result["aliases"]:
            if alias["alias"] == "test_fast":
                found = True
                break
        assert found, "Added alias should be in list"

        # Test save
        save_result = await tools.save_lockfile()
        assert save_result["success"] is True
        assert lockfile_path.exists()

        # Load and verify persistence
        lockfile = Lockfile.load(lockfile_path)
        resolved = lockfile.resolve_alias("test_fast")
        assert resolved is not None


@pytest.mark.asyncio
async def test_real_chat_app_initialization():
    """Test real MCPChatApp initialization without mocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "test.lock"

        # Create a minimal lockfile with advisor alias
        lockfile = Lockfile()
        lockfile.set_binding("advisor", "anthropic:claude-opus-4-1-20250805", profile="default")
        lockfile.save(lockfile_path)

        # Set environment to use our lockfile
        os.environ["LLMRING_LOCKFILE_PATH"] = str(lockfile_path)

        try:
            # Create real chat app
            app = MCPChatApp(
                mcp_server_url=None,  # No server for this test
                llm_model="advisor",
                enable_telemetry=False,
            )

            # Initialize without MCP server (tests lockfile loading)
            await app.initialize_async()

            # Verify advisor model is set
            # cmd_model is a method, model is the attribute
            assert hasattr(app, "model")
            assert app.model == "advisor"

            # Clean up
            await app.cleanup()

        finally:
            # Clean up environment
            if "LLMRING_LOCKFILE_PATH" in os.environ:
                del os.environ["LLMRING_LOCKFILE_PATH"]


@pytest.mark.asyncio
async def test_real_model_filtering():
    """Test real model filtering based on requirements."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "test.lock"

        tools = LockfileManagerTools(lockfile_path=lockfile_path)

        # Test filtering for coding (needs function calling)
        result = await tools.filter_models_by_requirements(
            requires_functions=True, min_context=50000
        )

        assert "models" in result
        assert len(result["models"]) > 0

        # Each model should meet requirements
        for model in result["models"]:
            assert model["supports_functions"] is True
            if model["context_window"]:
                assert model["context_window"] >= 50000

        # Test filtering for vision capabilities
        result2 = await tools.filter_models_by_requirements(requires_vision=True)

        assert "models" in result2
        # All should have vision support
        for model in result2["models"]:
            assert model["supports_vision"] is True


@pytest.mark.asyncio
async def test_real_cost_analysis():
    """Test real cost analysis without mocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "test.lock"

        tools = LockfileManagerTools(lockfile_path=lockfile_path)

        # Add some aliases
        await tools.add_alias("fast", "openai:gpt-4o-mini")
        await tools.add_alias("deep", "anthropic:claude-3-5-sonnet")

        # Analyze costs
        result = await tools.analyze_costs(
            monthly_volume={"input_tokens": 1000000, "output_tokens": 500000}
        )

        assert "total_monthly_cost" in result
        assert "cost_breakdown" in result
        assert isinstance(result["total_monthly_cost"], (int, float))

        # Should have costs for each alias (at least 1)
        assert len(result["cost_breakdown"]) >= 1


@pytest.mark.asyncio
async def test_real_model_assessment():
    """Test real model assessment without mocks."""
    tools = LockfileManagerTools()

    # Test assessment of real model
    result = await tools.assess_model("openai:gpt-4o")

    # If there's an error (e.g., registry not accessible), skip the test
    if "error" in result:
        pytest.skip(f"Model assessment failed: {result['error']}")

    assert "model" in result
    assert result["model"] == "openai:gpt-4o"  # Full model reference
    assert "capabilities" in result
    assert "active" in result

    # Test assessment of alias - aliases need full reference
    # Skip testing alias without full reference as it requires resolution


@pytest.mark.asyncio
async def test_real_profile_management():
    """Test real profile management without mocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "test.lock"

        tools = LockfileManagerTools(lockfile_path=lockfile_path)

        # Add to default profile
        await tools.add_alias(alias="fast", model="openai:gpt-4o-mini", profile="default")

        # Add to dev profile
        await tools.add_alias(alias="fast", model="anthropic:claude-3-haiku", profile="development")

        # List each profile
        default_result = await tools.list_aliases(profile="default")
        dev_result = await tools.list_aliases(profile="development")

        # Should have different models for same alias
        default_fast = next((a for a in default_result["aliases"] if a["alias"] == "fast"), None)
        dev_fast = next((a for a in dev_result["aliases"] if a["alias"] == "fast"), None)

        assert default_fast is not None
        assert dev_fast is not None
        assert default_fast["model"] != dev_fast["model"]


@pytest.mark.asyncio
async def test_real_error_handling():
    """Test real error handling without mocks."""
    tools = LockfileManagerTools()

    # Try to remove non-existent alias
    result = await tools.remove_alias("nonexistent_alias_xyz")

    assert result["success"] is False
    assert "message" in result or "error" in result

    # Try invalid model reference
    result2 = await tools.assess_model("invalid:model:format:too:many:colons")

    # Should handle gracefully
    assert "error" in result2 or "message" in result2


@pytest.mark.asyncio
async def test_real_lockfile_persistence():
    """Test real lockfile persistence without mocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "persist.lock"

        # First session - create configuration
        tools1 = LockfileManagerTools(lockfile_path=lockfile_path)

        await tools1.add_alias("coder", "anthropic:claude-3-5-sonnet")
        await tools1.add_alias("writer", "openai:gpt-4o")
        await tools1.save_lockfile()

        assert lockfile_path.exists()

        # Second session - load configuration
        tools2 = LockfileManagerTools(lockfile_path=lockfile_path)

        # Should load existing aliases
        result = await tools2.list_aliases()

        aliases = {a["alias"]: a["model"] for a in result["aliases"]}
        assert "coder" in aliases
        assert "writer" in aliases
        assert aliases["coder"] == "claude-3-5-sonnet"
        assert aliases["writer"] == "gpt-4o"


@pytest.mark.asyncio
async def test_real_configuration_export():
    """Test real configuration export without mocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "export.lock"

        # Load test lockfile to get real models
        test_lockfile_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "llmring.lock.json"
        )
        test_lockfile = Lockfile.load(Path(test_lockfile_path))

        tools = LockfileManagerTools(lockfile_path=lockfile_path)

        # Build configuration using existing aliases from test lockfile
        fast_model = test_lockfile.resolve_alias("fast")
        deep_model = test_lockfile.resolve_alias("deep")
        await tools.add_alias("config_fast", fast_model)
        await tools.add_alias("config_deep", deep_model)

        # Get full configuration
        config = await tools.get_current_configuration()

        assert "profiles" in config
        assert "metadata" in config
        assert "version" in config

        # Should have our aliases
        default_profile = config["profiles"]["default"]
        assert "bindings" in default_profile
        assert len(default_profile["bindings"]) >= 2


@pytest.mark.asyncio
async def test_real_data_driven_selection():
    """Test data-driven model selection using new tools."""
    tools = LockfileManagerTools()

    # Test various filtering scenarios
    test_cases = [
        # Quick chatbot - low cost
        {"max_price_input": 0.5, "min_context": 30000},
        # Code generation - function calling
        {"requires_functions": True, "min_context": 100000},
        # Document analysis - large context
        {"min_context": 200000},
        # Image description - vision
        {"requires_vision": True},
    ]

    for filters in test_cases:
        result = await tools.filter_models_by_requirements(**filters)

        assert "models" in result
        assert "applied_filters" in result

        # Verify filters were properly applied
        for model in result["models"]:
            if "min_context" in filters and model["context_window"]:
                assert model["context_window"] >= filters["min_context"]
            if "max_price_input" in filters and model["price_input"]:
                assert model["price_input"] <= filters["max_price_input"]
            if filters.get("requires_vision"):
                assert model["supports_vision"] is True
            if filters.get("requires_functions"):
                assert model["supports_functions"] is True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
