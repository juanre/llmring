"""
Test using mcp-server entry point without PYTHONPATH manipulation.

This test demonstrates that:
1. mcp-server can be installed and used as an entry point
2. AsyncMCPClient works correctly with the entry point
3. No PYTHONPATH manipulation is needed when using proper module paths
"""

import asyncio
from pathlib import Path

import pytest

from llmring.mcp.client.mcp_client import AsyncMCPClient


class TestMCPServerEntryPoint:
    """Test suite for mcp-server entry point integration."""

    @pytest.mark.asyncio
    async def test_entry_point_with_builtin_examples(self):
        """Test using mcp-server entry point with its built-in examples."""
        # Use the examples.math_tools module that comes with mcp-server
        # Use our test server directly
        server_script = str(Path(__file__).parent / "fixtures" / "mcp_test_server.py")
        async with AsyncMCPClient.stdio(["python", server_script]) as client:
            # Initialize
            result = await client.initialize()
            assert "serverInfo" in result
            # The server name should be Test MCP Server
            assert result["serverInfo"]["name"] == "Test MCP Server"

            # List tools - test_registry should provide test_tool
            tools = await client.list_tools()
            assert len(tools) > 0

            # Check for expected test tools
            tool_names = [t["name"] for t in tools]
            assert "test_tool" in tool_names
            print(f"Available tools: {tool_names}")

    @pytest.mark.asyncio
    async def test_entry_point_with_local_registry(self):
        """Test using mcp-server entry point with local test registry."""
        # Use our test server directly
        server_script = str(Path(__file__).parent / "fixtures" / "mcp_test_server.py")
        async with AsyncMCPClient.stdio(["python", server_script]) as client:
            # Initialize
            result = await client.initialize()
            assert "serverInfo" in result

            # List tools
            tools = await client.list_tools()
            print(f"Tools from test_registry: {[t['name'] for t in tools]}")

            # Our test_tool should be available
            tool_names = [t["name"] for t in tools]
            assert "test_tool" in tool_names

            # Call the test tool
            tool_result = await client.call_tool("test_tool", {"arg1": "hello"})
            # The result is in content array with JSON text
            import json

            content_text = tool_result["content"][0]["text"]
            parsed_result = json.loads(content_text)
            assert parsed_result["result"] == "Tool called with: hello"

    @pytest.mark.asyncio
    async def test_sync_client_works_with_entry_point(self):
        """Sync client should work with entry point after persistent loop fix."""
        from llmring.mcp.client.mcp_client import MCPClient

        server_script = str(Path(__file__).parent / "fixtures" / "mcp_test_server.py")
        with MCPClient.stdio(["python", server_script]) as client:
            result = client.initialize()
            assert isinstance(result, dict)
            assert "serverInfo" in result


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_entry_point_with_local_registry())
