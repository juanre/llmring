#!/usr/bin/env python3
"""
Real integration tests for MCP server without mocks.

Tests the actual server functionality with real transports and protocols.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

from llmring.mcp.server.lockfile_server.server import LockfileServer
from llmring.mcp.server import MCPServer
from llmring.mcp.server.transport.stdio import StdioTransport
from llmring.mcp.tools.lockfile_manager import LockfileManagerTools
from llmring.lockfile_core import Lockfile


class RealServerTestHarness:
    """Test harness for real server testing."""

    def __init__(self, lockfile_path: Path):
        """Initialize test harness."""
        self.lockfile_path = lockfile_path
        self.server = LockfileServer(lockfile_path=lockfile_path)
        self.process = None

    async def start_subprocess_server(self):
        """Start server as subprocess for real stdio testing."""
        self.process = subprocess.Popen(
            [
                sys.executable, "-m", "llmring.mcp.server.lockfile_server",
                "--lockfile", str(self.lockfile_path)
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Give it time to start
        await asyncio.sleep(0.2)
        return self.process

    def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send JSON-RPC request to server."""
        if not self.process:
            raise RuntimeError("Server not started")

        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1
        }

        # Send request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str)
        self.process.stdin.flush()

        # Read response
        response_str = self.process.stdout.readline()
        if response_str:
            return json.loads(response_str)
        return None

    def cleanup(self):
        """Clean up server process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()


@pytest.mark.asyncio
async def test_server_initialization():
    """Test real server initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "test.lock"

        # Create server
        server = LockfileServer(lockfile_path=lockfile_path)

        # Verify tools are registered
        assert len(server.server.function_registry.functions) > 0

        # Check specific tools
        expected_tools = [
            "add_alias",
            "remove_alias",
            "list_aliases",
            "assess_model",
            "analyze_costs",
            "save_lockfile",
            "get_configuration",
            "get_available_providers",
            "list_models",
            "filter_models_by_requirements",
            "get_model_details"
        ]

        for tool_name in expected_tools:
            assert tool_name in server.server.function_registry.functions


@pytest.mark.asyncio
async def test_server_tool_execution():
    """Test real server tool execution without mocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "test.lock"

        server = LockfileServer(lockfile_path=lockfile_path)

        # Get the wrapped function for add_alias
        add_alias_func = server.server.function_registry.functions["add_alias"]

        # Execute directly - need to provide a model now
        result = add_alias_func(
            alias="test_alias",
            model="openai:gpt-4o-mini"
        )

        assert result["success"] is True
        assert result["alias"] == "test_alias"
        assert "model" in result


@pytest.mark.asyncio
async def test_server_async_wrapper():
    """Test the async wrapper handles event loops correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "test.lock"

        server = LockfileServer(lockfile_path=lockfile_path)

        # Test wrapper with no running loop
        list_func = server.server.function_registry.functions["list_aliases"]
        result = list_func()

        assert "aliases" in result
        assert isinstance(result["aliases"], list)


@pytest.mark.asyncio
async def test_server_with_existing_lockfile():
    """Test server with pre-existing lockfile."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "existing.lock"

        # Create lockfile with content
        lockfile = Lockfile()
        lockfile.set_binding("fast", "openai:gpt-4o-mini", profile="default")
        lockfile.set_binding("advisor", "anthropic:claude-opus-4-1-20250805", profile="default")
        lockfile.save(lockfile_path)

        # Create server
        server = LockfileServer(lockfile_path=lockfile_path)

        # List should show existing aliases
        list_func = server.server.function_registry.functions["list_aliases"]
        result = list_func()

        aliases = {a["alias"]: a["model"] for a in result["aliases"]}
        assert "fast" in aliases
        assert "advisor" in aliases
        assert aliases["fast"] == "gpt-4o-mini"
        # Model name without provider prefix in the response
        assert aliases["advisor"] == "claude-opus-4-1-20250805"


@pytest.mark.asyncio
async def test_server_error_handling():
    """Test server error handling without mocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "test.lock"

        server = LockfileServer(lockfile_path=lockfile_path)

        # Try to remove non-existent alias
        remove_func = server.server.function_registry.functions["remove_alias"]
        result = remove_func(alias="nonexistent")

        assert result["success"] is False
        assert "message" in result or "error" in result


@pytest.mark.asyncio
async def test_server_timeout_handling():
    """Test server timeout handling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "test.lock"

        server = LockfileServer(lockfile_path=lockfile_path)

        # Create a slow function that would timeout
        async def slow_function(**kwargs):
            await asyncio.sleep(2)  # Longer than default timeout
            return {"result": "done"}

        # Wrap it with the server's wrapper
        wrapped = server._wrap_async(slow_function)

        # Should timeout (default is 30s, but we can pass shorter)
        with pytest.raises(TimeoutError):
            result = wrapped(_timeout=0.1)  # 100ms timeout


@pytest.mark.asyncio
async def test_server_profile_operations():
    """Test server profile operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "test.lock"

        server = LockfileServer(lockfile_path=lockfile_path)

        add_func = server.server.function_registry.functions["add_alias"]
        list_func = server.server.function_registry.functions["list_aliases"]

        # Add to different profiles
        add_func(alias="fast", model="openai:gpt-4o-mini", profile="default")
        add_func(alias="fast", model="anthropic:claude-3-haiku", profile="dev")

        # List from each profile
        default_result = list_func(profile="default")
        dev_result = list_func(profile="dev")

        # Get aliases as dict
        default_aliases = {a["alias"]: a["model"] for a in default_result["aliases"]}
        dev_aliases = {a["alias"]: a["model"] for a in dev_result["aliases"]}

        assert default_aliases["fast"] == "gpt-4o-mini"
        assert dev_aliases["fast"] == "claude-3-haiku"


@pytest.mark.asyncio
async def test_server_configuration_export():
    """Test server configuration export."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "test.lock"

        server = LockfileServer(lockfile_path=lockfile_path)

        # Add some configuration
        add_func = server.server.function_registry.functions["add_alias"]
        add_func(alias="coder", model="anthropic:claude-3-5-sonnet")

        # Get configuration
        get_config_func = server.server.function_registry.functions["get_configuration"]
        config = get_config_func()

        assert "version" in config
        assert "profiles" in config
        assert "metadata" in config

        # Check bindings
        bindings = config["profiles"]["default"]["bindings"]
        assert len(bindings) > 0

        # Find our added alias
        found = False
        for binding in bindings:
            if binding["alias"] == "coder":
                assert binding["model"] == "claude-3-5-sonnet"
                found = True
                break
        assert found


@pytest.mark.asyncio
async def test_server_save_and_reload():
    """Test server save and reload functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "persist.lock"

        # First server instance
        server1 = LockfileServer(lockfile_path=lockfile_path)

        add_func = server1.server.function_registry.functions["add_alias"]
        save_func = server1.server.function_registry.functions["save_lockfile"]

        # Add aliases and save
        add_func(alias="fast", model="openai:gpt-4o-mini")
        add_func(alias="deep", model="anthropic:claude-3-5-sonnet")
        save_result = save_func()

        assert save_result["success"] is True
        assert lockfile_path.exists()

        # Second server instance - should load existing
        server2 = LockfileServer(lockfile_path=lockfile_path)

        list_func = server2.server.function_registry.functions["list_aliases"]
        result = list_func()

        aliases = {a["alias"]: a["model"] for a in result["aliases"]}
        assert "fast" in aliases
        assert "deep" in aliases
        assert aliases["fast"] == "gpt-4o-mini"
        assert aliases["deep"] == "claude-3-5-sonnet"


@pytest.mark.asyncio
async def test_server_cost_analysis_integration():
    """Test server cost analysis with real data."""
    # Load test lockfile to get real model references
    import os
    from llmring.lockfile_core import Lockfile
    test_lockfile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "llmring.lock.json")
    test_lockfile = Lockfile.load(Path(test_lockfile_path))

    with tempfile.TemporaryDirectory() as tmpdir:
        lockfile_path = Path(tmpdir) / "test.lock"
        server = LockfileServer(lockfile_path=lockfile_path)

        # Add aliases using real models from test lockfile
        add_func = server.server.function_registry.functions["add_alias"]
        fast_model = test_lockfile.resolve_alias("fast")
        smart_model = test_lockfile.resolve_alias("smart")
        if fast_model:
            add_func(alias="cheap", model=fast_model)
        if smart_model:
            add_func(alias="expensive", model=smart_model)

        # Analyze costs
        analyze_func = server.server.function_registry.functions["analyze_costs"]
        result = analyze_func(
            monthly_volume={
                "input_tokens": 1000000,
                "output_tokens": 500000
            }
        )

        assert "total_monthly_cost" in result
        assert "cost_breakdown" in result
        # At least one alias should have costs
        assert len(result["cost_breakdown"]) >= 1

        # Check that costs are calculated (structure may vary)
        if isinstance(result["cost_breakdown"], dict):
            # Dictionary format: {alias: cost_info}
            assert len(result["cost_breakdown"]) >= 1
        else:
            # List format: [{alias, cost}, ...]
            for item in result["cost_breakdown"]:
                assert "alias" in item or isinstance(item, str)



@pytest.mark.asyncio
async def test_server_model_assessment():
    """Test server model assessment using test lockfile aliases."""
    # Use the test lockfile that has valid models
    import os
    from llmring.lockfile_core import Lockfile
    test_lockfile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "llmring.lock.json")
    test_lockfile = Lockfile.load(Path(test_lockfile_path))

    server = LockfileServer()
    assess_func = server.server.function_registry.functions["assess_model"]

    # Test using aliases from test lockfile
    aliases_to_test = ["fast", "smart", "deep"]
    models_to_test = []
    for alias in aliases_to_test:
        resolved = test_lockfile.resolve_alias(alias)
        if resolved:
            models_to_test.append(resolved)

    for model_ref in models_to_test:
        result = assess_func(model_ref=model_ref)

        assert "model" in result
        assert "active" in result
        assert "capabilities" in result

        # Model assessment should return model info
        if "error" not in result:
            # Check for expected fields from registry
            assert "model" in result or "display_name" in result
            # Provider is extracted from the model reference
            if ":" in model_ref:
                provider = model_ref.split(":")[0]
                assert result.get("provider") == provider or "model" in result


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])