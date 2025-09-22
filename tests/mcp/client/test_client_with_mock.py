"""
Test MCPClient with a mock transport to verify the client logic is correct.
"""

import pytest

from llmring.mcp.client.mcp_client import AsyncMCPClient, ConnectionState, MCPClient
from llmring.mcp.client.transports.base import Transport


class MockTransport(Transport):
    """Mock transport that simulates a compliant MCP server."""

    def __init__(self):
        super().__init__()
        self.sent_messages = []
        self.initialized = False

    async def start(self):
        """Start the mock transport."""
        self._set_state(ConnectionState.CONNECTED)

    async def send(self, message):
        """Send a message and return mock response."""
        self.sent_messages.append(message)
        method = message.get("method")

        if method == "initialize":
            return {
                "serverInfo": {
                    "name": "Mock MCP Server",
                    "version": "1.0.0",
                    "protocolVersion": "2025-03-26",
                },
                "capabilities": {
                    "tools": {"list": True, "call": True},
                    "resources": {"list": True, "read": True},
                    "prompts": {"list": True, "get": True},
                },
            }
        elif method == "tools/list":
            if not self.initialized:
                raise ValueError("JSON-RPC error -32000: Server not initialized")
            return [
                {"name": "test_tool", "description": "A test tool"},
                {"name": "another_tool", "description": "Another test tool"},
            ]
        elif method == "tools/call":
            if not self.initialized:
                raise ValueError("JSON-RPC error -32000: Server not initialized")
            name = message["params"]["name"]
            args = message["params"]["arguments"]
            return {"result": f"Called {name} with {args}"}

        return {}

    async def send_notification(self, message):
        """Handle notifications."""
        self.sent_messages.append(message)
        if message.get("method") == "initialized":
            self.initialized = True

    async def close(self):
        """Close the mock transport."""
        self._set_state(ConnectionState.DISCONNECTED)


def test_sync_client_with_mock():
    """Test sync MCPClient with mock transport."""
    transport = MockTransport()
    client = MCPClient(transport=transport)

    # Initialize
    result = client.initialize()
    assert "serverInfo" in result
    assert "capabilities" in result

    # Check that initialized notification was sent
    assert any(msg.get("method") == "initialized" for msg in transport.sent_messages)

    # List tools (should work now)
    tools = client.list_tools()
    assert len(tools) == 2
    assert tools[0]["name"] == "test_tool"

    # Call tool
    tool_result = client.call_tool("test_tool", {"arg": "value"})
    assert "result" in tool_result

    client.close()


@pytest.mark.asyncio
async def test_async_client_with_mock():
    """Test async MCPClient with mock transport."""
    transport = MockTransport()
    client = AsyncMCPClient(transport=transport)

    # Initialize
    result = await client.initialize()
    assert "serverInfo" in result
    assert "capabilities" in result

    # Check that initialized notification was sent
    assert any(msg.get("method") == "initialized" for msg in transport.sent_messages)

    # List tools (should work now)
    tools = await client.list_tools()
    assert len(tools) == 2
    assert tools[0]["name"] == "test_tool"

    # Call tool
    tool_result = await client.call_tool("test_tool", {"arg": "value"})
    assert "result" in tool_result

    await client.close()


if __name__ == "__main__":
    # Run sync test
    print("Testing sync client...")
    test_sync_client_with_mock()
    print("✓ Sync client test passed")

    # Run async test
    print("\nTesting async client...")
    import asyncio

    asyncio.run(test_async_client_with_mock())
    print("✓ Async client test passed")
