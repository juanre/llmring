"""
Test using AsyncMCPClient directly.
"""

import asyncio
import logging
from pathlib import Path

import pytest

from llmring.mcp.client.mcp_client import AsyncMCPClient

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)


TEST_SERVER = str(Path(__file__).parent / "fixtures" / "mcp_test_server.py")


@pytest.mark.asyncio
async def test_async_client_with_real_server():
    """Test AsyncMCPClient with real mcp-server."""
    # Create async client
    client = AsyncMCPClient.stdio(["python", TEST_SERVER])

    try:
        # Initialize
        print("Initializing...")
        result = await client.initialize()
        print(f"Initialize result: {result}")

        assert isinstance(result, dict)
        assert "serverInfo" in result
        assert "capabilities" in result

        # Check if server provides sessionInfo
        if "sessionInfo" in result:
            print(f"Session info: {result['sessionInfo']}")

        # List tools
        print("\nListing tools...")
        tools = await client.list_tools()
        print(f"Tools: {tools}")

        assert isinstance(tools, list)
        tool_names = [t.get("name") for t in tools]
        assert "test_tool" in tool_names

        # Call tool
        print("\nCalling test_tool...")
        tool_result = await client.call_tool("test_tool", {"arg1": "hello"})
        print(f"Tool result: {tool_result}")

        # Result is in content array with JSON text
        import json

        content_text = tool_result["content"][0]["text"]
        parsed_result = json.loads(content_text)
        assert parsed_result["result"] == "Tool called with: hello"

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_async_client_with_real_server())
