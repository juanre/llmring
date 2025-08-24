"""WebSocket transport implementation for MCP Server Engine."""

from llmring.mcp.server.transport.base import Transport


class WebSocketTransport(Transport):
    """WebSocket transport implementation."""

    async def start(self) -> bool:
        """Start the WebSocket transport."""
        raise NotImplementedError("WebSocket transport not yet implemented")

    async def stop(self) -> None:
        """Stop the WebSocket transport."""
        raise NotImplementedError("WebSocket transport not yet implemented")

    async def send_message(self, message: dict) -> bool:
        """Send a message through the WebSocket."""
        raise NotImplementedError("WebSocket transport not yet implemented")


class WebSocketServerTransport(WebSocketTransport):
    """WebSocket server transport implementation."""

    pass
